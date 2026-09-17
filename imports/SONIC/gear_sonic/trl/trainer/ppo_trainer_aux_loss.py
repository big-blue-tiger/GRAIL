import torch

from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer, masked_mean


class TRLAuxLossPPOTrainer(TRLPPOTrainer):
    """PPO trainer extended with per-step auxiliary losses.

    Subclasses :class:`TRLPPOTrainer` to add support for auxiliary losses
    that the policy's forward pass returns alongside the standard PPO
    objective.  Typical use-case is SONIC / universal token training where
    the ``UniversalTokenModule`` emits reconstruction and latent-alignment
    losses together with the action mean.

    The total loss is::

        loss = ppo_loss + aux_loss_scale * sum(coef_i * aux_loss_i)

    Auxiliary losses and their per-loss coefficients are expected in the
    ``policy_results`` dict under the keys ``"aux_losses"`` and
    ``"aux_loss_coef"`` respectively.

    Config keys (read from ``self.config``):

    * ``aux_loss_scale`` (float, default 1.0) – global scale applied to the
      weighted sum of auxiliary losses.
    * ``compute_aux_loss`` (bool, default ``True``) – disable to skip all
      auxiliary loss computation (useful for ablations).
    """

    _tag_names = ["trl", "aux_loss_ppo"]

    def _init_config(self):
        """Extend base config initialisation with auxiliary loss settings.

        Reads ``aux_loss_scale`` and ``compute_aux_loss`` from
        ``self.config`` and stores them as instance attributes.
        """
        super()._init_config()

        # Auxiliary loss configuration
        # aux_loss_scale: overall scalar to scale the total auxiliary loss
        self.aux_loss_scale = self.config.get("aux_loss_scale", 1.0)
        self.compute_aux_loss = self.config.get("compute_aux_loss", True)

        schedule_cfg = self.config.get("ppo_bc_loss_schedule", {})
        self.ppo_bc_loss_schedule_enabled = bool(schedule_cfg.get("enabled", False))
        self.ppo_bc_min_coef = float(schedule_cfg.get("bc_min_coef", 0.1))
        self.ppo_bc_decay_fraction = float(schedule_cfg.get("bc_decay_fraction", 0.5))
        self.ppo_bc_end_iteration = schedule_cfg.get("end_iteration", None)
        if self.ppo_bc_end_iteration is not None:
            self.ppo_bc_end_iteration = float(self.ppo_bc_end_iteration)
        if not 0.0 <= self.ppo_bc_min_coef <= 1.0:
            raise ValueError("ppo_bc_loss_schedule.bc_min_coef must be in [0, 1]")
        if not 0.0 < self.ppo_bc_decay_fraction <= 1.0:
            raise ValueError("ppo_bc_loss_schedule.bc_decay_fraction must be in (0, 1]")
        if self.ppo_bc_end_iteration is not None and self.ppo_bc_end_iteration <= 0:
            raise ValueError("ppo_bc_loss_schedule.end_iteration must be positive")

        contact_cfg = self.config.get("contact_loss_shift", {})
        self.contact_loss_shift_enabled = bool(contact_cfg.get("enabled", False))
        self.contact_loss_shift_amount = float(contact_cfg.get("amount", 0.05))
        self.contact_loss_command_name = contact_cfg.get("command_name", "motion")
        self.contact_loss_hand = contact_cfg.get("hand", "right_hand")
        if not 0.0 <= self.contact_loss_shift_amount <= 1.0:
            raise ValueError("contact_loss_shift.amount must be in [0, 1]")
        if self.contact_loss_hand not in ("left_hand", "right_hand"):
            raise ValueError("contact_loss_shift.hand must be left_hand or right_hand")
        if self.contact_loss_shift_enabled and (
            self.action_chunk_enabled or self.compute_imgaug_bc_loss or not self.compute_aux_loss
        ):
            raise ValueError("contact_loss_shift requires standard PPO with per-sample auxiliary BC")

    def _rollout_loss_contact_mask(self):
        if not self.contact_loss_shift_enabled:
            return None
        command = self.env.env.command_manager.get_term(self.contact_loss_command_name)
        in_contact = command.get_in_contact(self.contact_loss_hand)
        if in_contact is None:
            raise RuntimeError(
                f"contact_loss_shift requires reference contact labels for {self.contact_loss_hand}"
            )
        return (in_contact > 0.5).detach().clone()

    def _contact_loss_coefs(self, mb_rollout_data):
        """Shift scheduled coefficients per saved rollout label before reduction."""
        ppo, bc = self._get_ppo_bc_loss_coefs()
        mask = mb_rollout_data["loss_contact_mask"].float()
        shift = mask * self.contact_loss_shift_amount
        return (ppo + shift).clamp(0., 1.), (bc - shift).clamp(0., 1.)

    def _get_ppo_bc_loss_coefs(self):
        """Return the complementary PPO/BC coefficients for the current iteration.

        With the schedule enabled this implements

            lambda_bc(k) = max(bc_min_coef, 1 - k / (K * bc_decay_fraction))
            lambda_ppo(k) = 1 - lambda_bc(k)

        By default ``K * bc_decay_fraction`` defines the decay duration.  An
        explicit ``end_iteration`` overrides that duration, which is useful
        when reproducing a fixed curriculum from a paper.  Otherwise the
        existing static coefficients are returned.
        """
        if not self.ppo_bc_loss_schedule_enabled:
            return float(self.config.get("ppo_loss_coef", 1.0)), float(self.aux_loss_scale)

        iteration = float(self.state.global_step)
        if self.ppo_bc_end_iteration is not None:
            decay_iterations = self.ppo_bc_end_iteration
        else:
            total_iterations = float(self.args.num_total_batches)
            if total_iterations <= 0:
                raise ValueError("num_total_batches must be positive for PPO/BC loss scheduling")
            decay_iterations = total_iterations * self.ppo_bc_decay_fraction
        bc_coef = max(self.ppo_bc_min_coef, 1.0 - iteration / decay_iterations)
        ppo_coef = 1.0 - bc_coef
        return ppo_coef, bc_coef

    def _get_ppo_loss_coef(self, mb_rollout_data=None):
        """Return the scheduled PPO coefficient used by the base trainer."""
        if self.contact_loss_shift_enabled and mb_rollout_data is not None:
            return self._contact_loss_coefs(mb_rollout_data)[0]
        ppo_coef, _ = self._get_ppo_bc_loss_coefs()
        return ppo_coef

    def _register_stats_buffer(self):
        """Allocate per-step statistics tensors for auxiliary losses.

        Calls the parent method first to register base PPO stats, then
        allocates the following additional buffers when
        ``self.compute_aux_loss`` is ``True``:

        * ``self.aux_loss_stats`` – empty dict; individual loss tensors are
          added lazily on first occurrence (see :meth:`_update_stats_buffer`).
        * ``self.total_aux_loss_unscaled_stats`` – shape
          ``(num_ppo_epochs, num_mini_batches, num_micro_batches)``, stores
          the coefficient-weighted sum before the global scale.
        * ``self.total_aux_loss_stats`` – same shape, stores the fully scaled
          total auxiliary loss.
        """
        super()._register_stats_buffer()

        if self.compute_aux_loss:
            args = self.args
            device = self.accelerator.device

            stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.num_micro_batches)
            # Store stats as dictionaries to support multiple auxiliary losses
            self.aux_loss_stats = {}
            self.total_aux_loss_unscaled_stats = torch.zeros(stats_shape, device=device)
            self.total_aux_loss_stats = torch.zeros(stats_shape, device=device)
            self.effective_bc_coef_stats = torch.zeros(stats_shape, device=device)

    def _extract_aux_losses_from_forward_results(self, forward_results):
        """Pull auxiliary losses and their coefficients from the policy output dict.

        Args:
            forward_results: Dict returned by the policy's forward pass.  The
                following optional keys are consumed:

                * ``"aux_losses"`` – dict mapping loss name to scalar tensor.
                * ``"aux_loss_coef"`` – dict mapping loss name to float
                  coefficient; defaults to ``None`` when absent.

        Returns:
            Tuple of ``(aux_losses_dict, aux_loss_coef)`` where
            ``aux_losses_dict`` maps loss name to tensor and
            ``aux_loss_coef`` maps loss name to float (or ``None`` when the
            key was not present in ``forward_results``).
        """
        aux_losses_dict = {}
        aux_loss_coef = None

        if "aux_losses" in forward_results:
            aux_losses_dict = forward_results["aux_losses"]

        # Extract coefficients if provided in forward_results
        if "aux_loss_coef" in forward_results:
            aux_loss_coef = forward_results["aux_loss_coef"]

        return aux_losses_dict, aux_loss_coef

    def _compute_aux_loss(self, policy_results, mb_rollout_data):
        """Compute the weighted auxiliary loss for one mini-batch.

        Extracts individual auxiliary losses and their coefficients from
        ``policy_results``, computes the per-loss weighted sum, and applies
        the global ``aux_loss_scale``.

        Args:
            policy_results: Dict from the policy forward pass (the value
                stored at ``forward_results["policy_results"]``).  Expected
                to contain ``"aux_losses"`` and optionally ``"aux_loss_coef"``.
            mb_rollout_data: Mini-batch rollout dict (not used directly but
                available for subclass overrides).

        Returns:
            Dict with keys:

            * ``"aux_losses_dict"`` – raw loss tensors keyed by name.
            * ``"aux_loss_coef"`` – per-loss coefficient dict.
            * ``"total_aux_loss_unscaled"`` – weighted sum before
              ``aux_loss_scale``, shape ``()``.
            * ``"total_aux_loss"`` – final scaled auxiliary loss,
              shape ``()``.
        """
        device = self.accelerator.device

        # Extract auxiliary losses and coefficients from forward results
        aux_losses_dict, aux_loss_coef = self._extract_aux_losses_from_forward_results(
            policy_results
        )

        if not aux_losses_dict:
            if self.contact_loss_shift_enabled:
                raise RuntimeError("contact_loss_shift requires auxiliary BC losses")
            # No auxiliary losses found
            return {
                "aux_losses_dict": {},
                "aux_loss_coef": {},
                "total_aux_loss_unscaled": torch.tensor(0.0, device=device),
                "total_aux_loss": torch.tensor(0.0, device=device),
            }

        # Compute weighted sum of auxiliary losses
        total_aux_loss_unscaled = torch.tensor(0.0, device=device)
        for loss_name, loss_value in aux_losses_dict.items():
            coef = aux_loss_coef.get(loss_name, 0.0)
            total_aux_loss_unscaled += coef * loss_value

        # Apply either the static auxiliary scale or the scheduled BC weight.
        _, bc_coef = self._get_ppo_bc_loss_coefs()
        total_aux_loss = total_aux_loss_unscaled * bc_coef
        if self.contact_loss_shift_enabled:
            _, sample_bc_coef = self._contact_loss_coefs(mb_rollout_data)
            per_sample_losses = policy_results.get("aux_losses_per_sample", {})
            total_aux_loss = torch.tensor(0.0, device=device)
            for loss_name, loss_value in aux_losses_dict.items():
                coef = aux_loss_coef.get(loss_name, 0.0)
                if coef == 0.0:
                    continue
                if loss_name not in per_sample_losses:
                    raise RuntimeError(f"contact_loss_shift requires per-sample loss: {loss_name}")
                sample_loss = per_sample_losses[loss_name]
                if sample_loss.shape != sample_bc_coef.shape:
                    raise ValueError("BC loss and rollout contact labels must have identical shapes")
                total_aux_loss = total_aux_loss + coef * masked_mean(
                    sample_loss * sample_bc_coef, ~mb_rollout_data["mb_padding_mask"]
                )

        return {
            "aux_losses_dict": aux_losses_dict,
            "aux_loss_coef": aux_loss_coef,
            "total_aux_loss_unscaled": total_aux_loss_unscaled,
            "total_aux_loss": total_aux_loss,
            "effective_bc_coef": (
                masked_mean(sample_bc_coef, ~mb_rollout_data["mb_padding_mask"])
                if self.contact_loss_shift_enabled else bc_coef
            ),
        }

    def _compute_loss(self, forward_results, mb_rollout_data):
        """Compute total training loss as PPO loss plus auxiliary loss.

        Calls the parent ``_compute_loss`` for the standard PPO objective
        (and optionally imitation BC loss), then adds the auxiliary loss
        computed from ``forward_results["policy_results"]`` when
        ``self.compute_aux_loss`` is ``True``.

        Args:
            forward_results: Dict returned by the full forward pass.  Must
                contain ``"policy_results"`` (the policy module's output dict)
                plus whatever the parent method expects.
            mb_rollout_data: Mini-batch of rollout transitions used to
                compute advantages, returns, and old log-probs.

        Returns:
            Dict with at minimum:

            * ``"loss"`` – scalar total loss tensor (PPO + aux).
            * ``"aux_loss_dict"`` – result dict from
              :meth:`_compute_aux_loss` (only present when
              ``compute_aux_loss`` is ``True``).
            * All keys returned by the parent ``_compute_loss``.
        """
        # Compute PPO loss (includes ppo_loss and optionally imgaug_bc_loss)
        loss_dict = super()._compute_loss(forward_results, mb_rollout_data)

        # Compute and add auxiliary loss if enabled
        if self.compute_aux_loss:
            aux_loss_result = self._compute_aux_loss(
                forward_results["policy_results"], mb_rollout_data
            )

            loss_dict["loss"] += aux_loss_result["total_aux_loss"]

            # Add auxiliary loss dict to return dict
            loss_dict["aux_loss_dict"] = aux_loss_result

        return loss_dict

    def _update_stats_buffer(
        self,
        ppo_epoch_idx,
        minibatch_idx,
        microbatch_idx,
        loss_dict,
        forward_results,
        mb_rollout_data,
    ):
        """Record per-step loss values into pre-allocated statistics buffers.

        Delegates to the parent method for base PPO stats, then writes
        individual auxiliary loss values and aggregate totals into
        ``self.aux_loss_stats``, ``self.total_aux_loss_unscaled_stats``, and
        ``self.total_aux_loss_stats``.  Individual loss buffers are lazily
        initialised on first encounter.

        Args:
            ppo_epoch_idx: Index of the current PPO epoch (0-based).
            minibatch_idx: Index of the current mini-batch within the epoch.
            microbatch_idx: Index of the current micro-batch within the
                mini-batch (used for gradient accumulation).
            loss_dict: Output of :meth:`_compute_loss` for this step.
                Expected to contain ``"aux_loss_dict"`` when
                ``compute_aux_loss`` is ``True``.
            forward_results: Full forward-pass result dict (passed through to
                the parent method).
            mb_rollout_data: Mini-batch rollout dict (passed through to the
                parent method).
        """
        # Update PPO stats
        super()._update_stats_buffer(
            ppo_epoch_idx,
            minibatch_idx,
            microbatch_idx,
            loss_dict,
            forward_results,
            mb_rollout_data,
        )

        # Update auxiliary loss stats if enabled
        if self.compute_aux_loss and "aux_loss_dict" in loss_dict:
            aux_loss_result = loss_dict["aux_loss_dict"]
            aux_losses_dict = aux_loss_result["aux_losses_dict"]
            total_aux_loss_unscaled = aux_loss_result["total_aux_loss_unscaled"]
            total_aux_loss = aux_loss_result["total_aux_loss"]

            # Update stats for each individual auxiliary loss
            for loss_name, loss_value in aux_losses_dict.items():
                if loss_name not in self.aux_loss_stats:
                    # Lazily initialize stats buffer for this loss
                    args = self.args
                    device = self.accelerator.device
                    stats_shape = (
                        args.num_ppo_epochs,
                        args.num_mini_batches,
                        args.num_micro_batches,
                    )
                    self.aux_loss_stats[loss_name] = torch.zeros(stats_shape, device=device)

                self.aux_loss_stats[loss_name][
                    ppo_epoch_idx, minibatch_idx, microbatch_idx
                ] = loss_value

            # Update total auxiliary loss stats
            self.total_aux_loss_unscaled_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = (
                total_aux_loss_unscaled
            )
            self.total_aux_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = total_aux_loss
            self.effective_bc_coef_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = (
                aux_loss_result.get("effective_bc_coef", 0.0)
            )

    def _get_train_metrics(self):
        """Collect training metrics including auxiliary loss averages.

        Calls the parent method for standard PPO metrics, then appends:

        * ``"loss/aux_{name}_avg"`` – mean of each individual auxiliary loss
          across all PPO epochs, mini-batches, and micro-batches.
        * ``"loss/total_aux_loss_unscaled_avg"`` – mean of the
          coefficient-weighted sum before global scaling.
        * ``"loss/total_aux_loss_avg"`` – mean of the fully scaled total
          auxiliary loss.
        * ``"aux_loss_scale"`` – the configured global scale factor.

        Returns:
            Dict of metric name → scalar value for the completed training
            iteration, suitable for logging to W&B or TensorBoard.
        """
        metrics = super()._get_train_metrics()

        # Add auxiliary loss metrics if enabled
        if self.compute_aux_loss:
            # Add metrics for each individual auxiliary loss
            for loss_name, loss_stats in self.aux_loss_stats.items():
                metrics[f"loss/aux_{loss_name}_avg"] = (
                    self.accelerator.gather_for_metrics(loss_stats).mean().item()
                )

            # Add total auxiliary loss metrics
            metrics["loss/total_aux_loss_unscaled_avg"] = (
                self.accelerator.gather_for_metrics(self.total_aux_loss_unscaled_stats)
                .mean()
                .item()
            )
            metrics["loss/total_aux_loss_avg"] = (
                self.accelerator.gather_for_metrics(self.total_aux_loss_stats).mean().item()
            )
            metrics["aux_loss_scale"] = self.aux_loss_scale
            metrics["loss/effective_bc_coef"] = (
                self.accelerator.gather_for_metrics(self.effective_bc_coef_stats).mean().item()
            )

        ppo_coef, bc_coef = self._get_ppo_bc_loss_coefs()
        metrics["loss/lambda_ppo"] = ppo_coef
        metrics["loss/lambda_bc"] = bc_coef

        return metrics
