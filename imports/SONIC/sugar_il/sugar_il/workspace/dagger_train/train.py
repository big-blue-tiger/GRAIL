#!/usr/bin/env python3
"""Train the object-aware flow generator with GPU-resident online DAgger."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import copy
import gc
import math
import os
from pathlib import Path
import random
import sys
import time
import faulthandler

import dill
import numpy as np
import torch
import tqdm


def _parser(app_launcher_cls) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="DAgger YAML config; remaining key=value arguments override it",
    )
    app_launcher_cls.add_app_launcher_args(parser)
    return parser


def _pickle(payload: dict, key: str, default=None):
    value = payload.get("pickles", {}).get(key)
    return default if value is None else dill.loads(value)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_states = state.get("cuda")
    if not cuda_states or not torch.cuda.is_available():
        return
    visible_devices = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    if len(cuda_states) != visible_devices:
        print(
            "[dagger] resume CUDA RNG device count differs: "
            f"checkpoint={len(cuda_states)}, visible={visible_devices}; "
            f"restoring device={current_device} only",
            flush=True,
        )
    if current_device < len(cuda_states):
        torch.cuda.set_rng_state(cuda_states[current_device], device=current_device)


def _copy_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _copy_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_cpu(item) for item in value)
    return copy.deepcopy(value)


def _teacher_mask(num_envs: int, ratio: float, device: torch.device) -> torch.Tensor:
    count = int(num_envs * ratio)
    mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    mask[torch.randperm(num_envs, device=device)[:count]] = True
    return mask


def _termination_reasons(base_env, env_id: int) -> tuple[str, ...]:
    manager = base_env.termination_manager
    values = getattr(manager, "_last_episode_dones", None)
    if values is None:
        values = getattr(manager, "_term_dones", None)
    if values is None:
        return ()
    return tuple(
        name
        for name, triggered in zip(manager.active_terms, values[env_id].tolist(), strict=True)
        if triggered
    )


def _last(value: torch.Tensor) -> torch.Tensor:
    return value[:, -1] if value.ndim == 3 else value


def _cuda_status() -> str:
    """Return lightweight CUDA telemetry without forcing a device sync."""
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    try:
        return (
            f"device={torch.cuda.current_device()}"
            f", allocated={torch.cuda.memory_allocated() / 2**30:.2f}GiB"
            f", reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB"
        )
    except Exception as exc:  # pragma: no cover - CUDA runtime dependent
        return f"cuda_status_error={type(exc).__name__}: {exc}"


def _flow_stage_start(label: str) -> float:
    started = time.monotonic()
    print(f"[dagger] flow init: {label}...", flush=True)
    return started


def _flow_stage_done(label: str, started: float) -> None:
    elapsed = time.monotonic() - started
    print(
        f"[dagger] flow init: {label} complete in {elapsed:.2f}s ({_cuda_status()})",
        flush=True,
    )


def _create_accelerator(cfg):
    """Create the one accelerator before Isaac Sim initializes its CUDA context."""
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration

    project = ProjectConfiguration(
        project_dir=str(cfg.paths.output_dir),
        logging_dir=str(Path(cfg.paths.output_dir) / "tb"),
    )
    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        log_with="tensorboard",
        project_config=project,
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("Online DAgger currently supports one GPU only")
    if accelerator.device.type == "cuda":
        torch.cuda.set_device(accelerator.local_process_index)
    print(
        f"[dagger] accelerator ready: device={accelerator.device}, "
        f"processes={accelerator.num_processes} ({_cuda_status()})",
        flush=True,
    )
    return accelerator


def _sim_attr(sim, name: str):
    value = getattr(sim, name, None)
    return value() if callable(value) else value


def _check_render_state(env) -> None:
    sim = env.env.sim
    render_mode = getattr(sim, "render_mode", None)
    has_gui = _sim_attr(sim, "has_gui")
    if has_gui is None:
        has_gui = getattr(sim, "_has_gui", None)
    render_mode_name = getattr(render_mode, "name", None) or repr(render_mode)
    if has_gui is None and any(
        name in render_mode_name for name in ("NO_GUI_OR_RENDERING", "PARTIAL_RENDERING")
    ):
        has_gui = False
    has_rtx_sensors = _sim_attr(sim, "has_rtx_sensors")
    print(
        "[dagger] simulation render state: "
        f"mode={render_mode!r}, has_gui={has_gui!r}, "
        f"has_rtx_sensors={has_rtx_sensors!r}, "
        f"ENABLE_CAMERAS={os.environ.get('ENABLE_CAMERAS', '<unset>')}",
        flush=True,
    )
    if has_gui is True or has_rtx_sensors is True:
        raise RuntimeError(
            "DAgger requires headless simulation without GUI or RTX sensors; "
            f"got mode={render_mode!r}, has_gui={has_gui!r}, "
            f"has_rtx_sensors={has_rtx_sensors!r}"
        )


class DaggerTrainer:
    def __init__(self, cfg, env, teacher, simulation_app, initial_obs, accelerator) -> None:
        import hydra

        from sugar_il.model.common.lr_scheduler import get_scheduler
        from sugar_il.workspace.dagger_train.storage import FlowRolloutStorage

        self.cfg = cfg
        self.env = env
        self.teacher = teacher
        self.simulation_app = simulation_app
        self.output_dir = Path(cfg.paths.output_dir)
        if accelerator.num_processes != 1:
            raise RuntimeError("Online DAgger currently supports one GPU only")
        self.accelerator = accelerator

        started = _flow_stage_start("loading checkpoint on CPU")
        self.payload = torch.load(
            cfg.paths.generator_checkpoint,
            map_location="cpu",
            pickle_module=dill,
            weights_only=False,
        )
        _flow_stage_done("checkpoint loaded on CPU", started)
        states = self.payload.get("state_dicts", {})
        required = {"model", "optimizer", "lr_scheduler"}
        if missing := required.difference(states):
            raise KeyError("Resume checkpoint is missing: " + ", ".join(sorted(missing)))

        self.flow_cfg = self.payload["cfg"]
        started = _flow_stage_start("instantiating and restoring flow model on CPU")
        self.model = hydra.utils.instantiate(self.flow_cfg.policy)
        self.model.load_state_dict(states["model"])
        if not self.model.normalizer.params_dict:
            raise ValueError("Flow checkpoint does not contain a fitted normalizer")

        self.ema_model = None
        self.ema = None
        use_ema = bool(self.flow_cfg.training.get("use_ema", False))
        if use_ema:
            if "ema_model" not in states:
                raise KeyError("EMA-enabled checkpoint is missing ema_model")
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.load_state_dict(states["ema_model"])
            self.ema = hydra.utils.instantiate(self.flow_cfg.ema, model=self.ema_model)
            self.ema.optimization_step = _pickle(self.payload, "ema_optimization_step", 0)
        _flow_stage_done("flow model restored on CPU", started)

        started = _flow_stage_start("initializing TensorBoard tracker")
        self.accelerator.init_trackers(cfg.logging.project)
        _flow_stage_done("TensorBoard tracker initialized", started)

        started = _flow_stage_start("restoring optimizer and scheduler on CPU")
        self.optimizer = self.model.get_optimizer(**self.flow_cfg.optimizer)
        total_steps = _pickle(self.payload, "scheduler_total_steps")
        if total_steps is None:
            raise KeyError("Resume checkpoint is missing scheduler_total_steps")
        self.scheduler = get_scheduler(
            self.flow_cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.flow_cfg.training.lr_warmup_steps,
            num_training_steps=total_steps,
            last_epoch=-1,
        )
        self.optimizer.load_state_dict(states["optimizer"])
        self.scheduler.load_state_dict(states["lr_scheduler"])
        _flow_stage_done("optimizer and scheduler restored on CPU", started)

        started = _flow_stage_start("preparing flow model and optimizer on CUDA")
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        _flow_stage_done("flow model and optimizer prepared on CUDA", started)
        device = self.accelerator.device
        if self.ema_model is not None:
            started = _flow_stage_start("moving EMA model to CUDA")
            self.ema_model.to(device)
            _flow_stage_done("EMA model moved to CUDA", started)

        started = _flow_stage_start("restoring DAgger state and RNG")
        self.dagger_state = _pickle(self.payload, "dagger_state", {})
        self.iteration = int(self.dagger_state.get("iteration", 0))
        self.global_step = int(_pickle(self.payload, "global_step", 0))
        saved_runtime = self.dagger_state.get("runtime")
        runtime = {
            "teacher_ratio": cfg.training.teacher_ratio,
            "epochs_per_rollout": cfg.training.epochs_per_rollout,
            "batch_size": cfg.training.batch_size,
            "max_parallel_envs": cfg.environment.max_parallel_envs,
            "input": str(cfg.paths.input),
            "teacher_checkpoint": str(cfg.paths.teacher_checkpoint),
        }
        if saved_runtime is not None:
            # The number of simulator environments only controls rollout
            # parallelism.  It is safe to change between runs, and allowing
            # it lets a checkpoint created with a larger batch resume on a
            # GPU with less PhysX memory.  Keep the data/model-affecting
            # settings strict so an accidental resume still fails loudly.
            mismatches = {
                key: (saved_runtime.get(key), value)
                for key, value in runtime.items()
                if saved_runtime.get(key) != value
            }
            parallelism = mismatches.pop("max_parallel_envs", None)
            if parallelism is not None:
                print(
                    "[dagger] resume changing max_parallel_envs: "
                    f"checkpoint={parallelism[0]}, current={parallelism[1]}",
                    flush=True,
                )
            if mismatches:
                details = ", ".join(
                    f"{key}={old!r}->{new!r}"
                    for key, (old, new) in sorted(mismatches.items())
                )
                raise ValueError(
                    "DAgger resume configuration differs from its checkpoint: "
                    + details
                )
        self.runtime = runtime
        _restore_rng(_pickle(self.payload, "rng_state"))
        _flow_stage_done("DAgger state and RNG restored", started)

        started = _flow_stage_start("moving initial observations to accelerator device")
        obs = {
            key: value.to(self.accelerator.device) for key, value in initial_obs.items()
        }
        self._initial_obs = obs
        _flow_stage_done("initial observations moved to accelerator device", started)
        self.num_unique_motions = int(env._motion_lib._num_unique_motions)  # noqa: SLF001
        self.num_motion_batches = math.ceil(self.num_unique_motions / env.num_envs)
        self._loaded_motion_batch = 0
        from isaaclab.managers import SceneEntityCfg
        from gear_sonic.envs.manager_env.mdp.observations import hand_object_transform_6d
        from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper
        from sugar_il.workspace.dagger_train.environment import generator_observation

        self.hand_frame_cfg = SceneEntityCfg("object_to_hand_frame_transformer")
        self.hand_transform_fn = hand_object_transform_6d
        rollout_model = self.ema_model or self.accelerator.unwrap_model(self.model)
        started = _flow_stage_start("building initial generator observation")
        wrapper = GeneratorWrapper(rollout_model, device)
        generator_obs = generator_observation(
            env, wrapper, obs, self.hand_frame_cfg, self.hand_transform_fn
        )
        _flow_stage_done("initial generator observation built", started)
        capacity = max(
            int(env.motion_command.motion_num_steps.max().item()) + 1,
            int(getattr(env.env, "max_episode_length", 0)) + 1,
        )
        shapes = {key: tuple(value.shape[1:]) for key, value in generator_obs.items()}
        started = _flow_stage_start("allocating rollout storage")
        self.storage = FlowRolloutStorage(
            env.num_envs,
            capacity,
            shapes,
            horizon=cfg.rollout.horizon,
            device=device,
        )
        self.batch_size = cfg.training.batch_size or int(self.flow_cfg.dataloader.batch_size)
        _flow_stage_done("rollout storage allocated", started)

    def _reset(self) -> dict[str, torch.Tensor]:
        print("[dagger] resetting environments for a new rollout", flush=True)
        started = time.monotonic()
        obs = self.env.reset_all()
        print(
            f"[dagger] rollout reset complete in {time.monotonic() - started:.1f}s",
            flush=True,
        )
        starts = self.env.motion_command.motion_start_time_steps
        if torch.any(starts != 0):
            raise RuntimeError("Every DAgger episode must start from motion frame zero")
        return {key: value.to(self.accelerator.device) for key, value in obs.items()}

    def _load_motion_batch(self, batch_index: int) -> dict[str, torch.Tensor]:
        """Load one sequential motion batch and reset it at frame zero."""
        start_idx = batch_index * self.env.num_envs
        end_idx = min(start_idx + self.env.num_envs, self.num_unique_motions)
        print(
            f"[dagger] loading motion batch {batch_index + 1}/"
            f"{self.num_motion_batches}: motions {start_idx}:{end_idx}",
            flush=True,
        )
        self.env.start_idx = start_idx
        self.env._motion_lib.load_motions_for_evaluation(start_idx=start_idx)  # noqa: SLF001
        self._loaded_motion_batch = batch_index
        return self._reset()

    @torch.no_grad()
    def _teacher_targets(self, obs_dict) -> tuple[torch.Tensor, torch.Tensor]:
        # SONIC evaluation resets the one-step rollout buffer before every
        # policy call. Keeping the previous TensorDict makes the second call
        # concatenate every observation before trimming max_rollout_history,
        # which is prohibitively expensive for many parallel environments.
        trace = getattr(self, "_phase_trace", False)
        if trace:
            print("[dagger] teacher: init_rollout start", flush=True)
        self.teacher.init_rollout()
        if trace:
            print("[dagger] teacher: init_rollout complete", flush=True)
            print("[dagger] teacher: policy rollout start", flush=True)
        self.teacher.rollout(obs_dict=obs_dict)
        if trace:
            print("[dagger] teacher: policy rollout complete", flush=True)
        residual = self.teacher.action_mean[:, :64].detach()
        scale = float(self.env._latent_residual_scale)  # noqa: SLF001
        mode = self.env._latent_residual_mode  # noqa: SLF001
        if mode != "pre_quantization" or abs(scale - 0.1) > 1e-8:
            raise ValueError("Teacher requires pre_quantization residual scale 0.1")
        atm_obs = self.env._prepare_obs_for_action_transform_module(obs_dict)  # noqa: SLF001
        if trace:
            print("[dagger] teacher: ATM forward start", flush=True)
        self.env.action_transform_module(
            atm_obs,
            latent_residual=residual * scale,
            latent_residual_mode="pre_quantization",
        )
        if trace:
            print("[dagger] teacher: ATM forward complete", flush=True)
        latent = _last(
            self.env.action_transform_module.actor_module._last_pre_quantization_latent_flat  # noqa: SLF001
        ).detach()
        if trace:
            print("[dagger] teacher: latent extraction complete", flush=True)
        left = self.env.motion_command.get_hand_action("left_hand")
        right = self.env.motion_command.get_hand_action("right_hand")
        if left is None or right is None:
            raise ValueError("Reference motion is missing hand actions")
        hand = (torch.stack((left, right), dim=-1) >= 0).to(latent.dtype)
        if trace:
            print("[dagger] teacher: hand extraction complete", flush=True)
        return latent, hand

    def collect(self):
        from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper
        from sugar_il.workspace.dagger_train.environment import (
            binary_hand_to_sonic,
            generator_observation,
        )

        device = self.accelerator.device
        motion_batch = self.iteration % self.num_motion_batches
        if self._initial_obs is not None and motion_batch == self._loaded_motion_batch:
            obs_dict = self._initial_obs
            self._initial_obs = None
        elif motion_batch != self._loaded_motion_batch:
            self._initial_obs = None
            obs_dict = self._load_motion_batch(motion_batch)
        else:
            obs_dict = self._reset()
        self.storage.clear()
        active = torch.ones(self.env.num_envs, dtype=torch.bool, device=device)
        active_count = self.env.num_envs
        teacher_execution = _teacher_mask(
            self.env.num_envs, self.cfg.training.teacher_ratio, device
        )
        episode_starts = active.clone()
        reasons: Counter[str] = Counter()
        time_out_count = 0
        lengths = torch.zeros(self.env.num_envs, dtype=torch.long, device=device)
        horizon = self.cfg.rollout.horizon
        execution_horizon = self.cfg.rollout.execution_horizon
        plan_latent = torch.zeros(self.env.num_envs, horizon, 64, device=device)
        plan_hand = torch.zeros(self.env.num_envs, horizon, 2, device=device)
        plan_cursor = torch.full(
            (self.env.num_envs,), execution_horizon, dtype=torch.long, device=device
        )
        rollout_model = self.ema_model or self.accelerator.unwrap_model(self.model)
        wrapper = GeneratorWrapper(rollout_model, device)
        progress_every = int(self.cfg.rollout.progress_every)
        stall_trace_seconds = int(self.cfg.rollout.stall_trace_seconds)
        phase_trace = bool(self.cfg.rollout.get("phase_trace", False))
        self._phase_trace = phase_trace
        started_at = time.monotonic()
        teacher_count = int(teacher_execution.sum().item())
        print(
            f"[dagger] rollout {self.iteration + 1} started: "
            f"envs={self.env.num_envs}, teacher={teacher_count}, "
            f"student={self.env.num_envs - teacher_count}, "
            f"motion_batch={motion_batch + 1}/{self.num_motion_batches}, "
            f"capacity={self.storage.num_transitions_per_env}",
            flush=True,
        )

        if stall_trace_seconds > 0:
            faulthandler.dump_traceback_later(stall_trace_seconds, repeat=True)

        while active_count > 0 and self.simulation_app.is_running():
            if self.storage.step >= self.storage.num_transitions_per_env:
                raise RuntimeError("Rollout exceeded the longest reference motion")
            step_number = self.storage.step + 1

            def trace_phase(label: str) -> None:
                if phase_trace:
                    print(f"[dagger] step {step_number}: {label}", flush=True)

            trace_step = self.storage.step < 3
            trace_phase("generator observation start")
            if trace_step:
                print(
                    f"[dagger] step {self.storage.step + 1}: "
                    "building generator observation",
                    flush=True,
                )
            generator_obs = generator_observation(
                self.env,
                wrapper,
                obs_dict,
                self.hand_frame_cfg,
                self.hand_transform_fn,
            )
            trace_phase("generator observation complete")
            trace_phase("teacher targets start")
            if trace_step:
                print(
                    f"[dagger] step {self.storage.step + 1}: running teacher policy",
                    flush=True,
                )
            teacher_latent, teacher_hand = self._teacher_targets(obs_dict)
            trace_phase("teacher targets complete")
            refresh = active & ~teacher_execution & (plan_cursor >= execution_horizon)
            if refresh.any():
                ids = refresh.nonzero(as_tuple=True)[0]
                if trace_step:
                    print(
                        f"[dagger] step {self.storage.step + 1}: "
                        f"planning for {len(ids)} student environments",
                        flush=True,
                    )
                trace_phase(f"student prediction start ({len(ids)} envs)")
                prediction = rollout_model.predict_action(
                    {key: value[ids] for key, value in generator_obs.items()}
                )
                trace_phase("student prediction complete")
                plan_latent[ids] = prediction["latent"]
                plan_hand[ids] = prediction["hand_primitive"]
                plan_cursor[ids] = 0

            student_ids = (active & ~teacher_execution).nonzero(as_tuple=True)[0]
            action_latent = teacher_latent.clone()
            action_hand = teacher_hand.clone()
            if len(student_ids):
                cursor = plan_cursor[student_ids]
                action_latent[student_ids] = plan_latent[student_ids, cursor]
                action_hand[student_ids] = plan_hand[student_ids, cursor]
                plan_cursor[student_ids] += 1
            action = torch.cat((action_latent, binary_hand_to_sonic(action_hand)), dim=-1)
            if trace_step:
                print(
                    f"[dagger] step {self.storage.step + 1}: "
                    "stepping IsaacLab environments",
                    flush=True,
                )
            trace_phase("environment step start")
            next_obs, _, dones, infos = self.env.step(
                {"actions": action, "obs_dict": obs_dict, "action_mode": "direct_latent"}
            )
            trace_phase("environment step complete")
            dones = dones.reshape(-1).bool().to(device)
            time_outs = infos["time_outs"].reshape(-1).bool().to(device)
            trace_phase("storage append start")
            self.storage.append(
                generator_obs,
                teacher_latent,
                teacher_hand,
                active,
                dones & active,
                time_outs & active,
                episode_starts,
                teacher_execution,
            )
            trace_phase("storage append complete")
            if trace_step:
                print(
                    f"[dagger] step {self.storage.step}: completed",
                    flush=True,
                )
            if stall_trace_seconds > 0:
                # Rearm after every successful environment step so a traceback
                # means one phase truly stopped returning, not merely that the
                # complete rollout took longer than the timeout.
                faulthandler.cancel_dump_traceback_later()
                faulthandler.dump_traceback_later(stall_trace_seconds, repeat=True)
            lengths += active.long()
            finished = dones & active
            finished_ids = finished.nonzero(as_tuple=True)[0].tolist()
            for env_id in finished_ids:
                term_reasons = _termination_reasons(self.env.env, env_id)
                reasons.update(term_reasons or ("unknown",))
            time_out_count += int((finished & time_outs).sum().item())
            active_count -= len(finished_ids)
            active &= ~finished
            trace_phase("done processing complete")
            episode_starts.zero_()
            obs_dict = {key: value.to(device) for key, value in next_obs.items()}
            if self.storage.step % progress_every == 0 or active_count == 0:
                elapsed = time.monotonic() - started_at
                print(
                    f"[dagger] rollout step={self.storage.step}/"
                    f"{self.storage.num_transitions_per_env}, active={active_count}/"
                    f"{self.env.num_envs}, elapsed={elapsed:.1f}s, "
                    f"steps_per_second={self.storage.step / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )

        # Do not call Actor.clear_rollout() directly after the final Isaac
        # step. It drops CUDA-backed distribution/history tensors and can
        # synchronize with Kit's CUDA context indefinitely. The next teacher
        # call starts with init_rollout(), and process teardown owns the final
        # release.
        print(
            "[dagger] rollout loop exited; retaining teacher CUDA state for "
            f"deferred cleanup ({_cuda_status()})",
            flush=True,
        )
        active_started = time.monotonic()
        active_remaining = active_count > 0
        print(
            f"[dagger] active check complete in "
            f"{time.monotonic() - active_started:.3f}s, remaining={active_remaining} "
            f"({_cuda_status()})",
            flush=True,
        )
        if active_remaining:
            if stall_trace_seconds > 0:
                faulthandler.cancel_dump_traceback_later()
            return None
        counts_started = time.monotonic()
        total, teacher_windows, student_windows = self.storage.window_counts()
        print(
            f"[dagger] window counts complete in "
            f"{time.monotonic() - counts_started:.3f}s ({_cuda_status()})",
            flush=True,
        )
        if stall_trace_seconds > 0:
            faulthandler.cancel_dump_traceback_later()
        print(
            f"[dagger] rollout complete: windows={total}, "
            f"teacher_windows={teacher_windows}, student_windows={student_windows}",
            flush=True,
        )
        return {
            "windows": total,
            "teacher_windows": teacher_windows,
            "student_windows": student_windows,
            "teacher_ratio": float(teacher_execution.float().mean().item()),
            "mean_episode_length": float(lengths.float().mean().item()),
            "time_out_rate": time_out_count / self.env.num_envs,
            "early_termination_rate": 1 - time_out_count / self.env.num_envs,
            "termination_reasons": dict(reasons),
        }

    def train_latest(self) -> dict[str, float]:
        total, _, _ = self.storage.window_counts()
        if not total:
            raise RuntimeError("Latest rollout contains no valid 40-step training windows")
        # Rollout inference leaves temporary CUDA blocks in the caching
        # allocator. Release those before allocating activations and gradients.
        gc.collect()
        if torch.cuda.is_available():
            cache_started = time.monotonic()
            print(
                f"[dagger] training cache clear starting ({_cuda_status()})",
                flush=True,
            )
            torch.cuda.empty_cache()
            print(
                f"[dagger] training cache clear complete in "
                f"{time.monotonic() - cache_started:.3f}s ({_cuda_status()})",
                flush=True,
            )
        model = self.model
        model.train()
        normalization_started = time.monotonic()
        self.storage.normalize_(self.accelerator.unwrap_model(model).normalizer)
        normalization_seconds = time.monotonic() - normalization_started
        accumulation = int(self.flow_cfg.training.gradient_accumulate_every)
        epochs = self.cfg.training.epochs_per_rollout
        num_batches = math.ceil(total / self.batch_size) * epochs
        progress_every = int(self.cfg.training.progress_every)
        stall_trace_seconds = int(self.cfg.rollout.stall_trace_seconds)
        memory = ""
        if torch.cuda.is_available():
            memory = (
                f", cuda_allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB"
                f", cuda_reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB"
            )
        print(
            f"[dagger] training started: windows={total}, batch_size={self.batch_size}, "
            f"batches={num_batches}, accumulation={accumulation}, "
            f"normalize_once={normalization_seconds:.3f}s{memory}",
            flush=True,
        )
        training_started_at = time.monotonic()
        sums = Counter()
        self.optimizer.zero_grad()
        if stall_trace_seconds > 0:
            faulthandler.dump_traceback_later(stall_trace_seconds, repeat=True)
        batches = self.storage.batches(self.batch_size, epochs)
        with tqdm.tqdm(
            batches,
            total=num_batches,
            desc=f"DAgger training iteration {self.iteration + 1}",
            leave=True,
            mininterval=1.0,
        ) as train_progress:
            for batch_idx, batch in enumerate(train_progress):
                trace_batch = batch_idx < 3
                if trace_batch:
                    batch_shapes = ", ".join(
                        f"{key}={tuple(value.shape)}"
                        for key, value in batch.get("obs", {}).items()
                    )
                    print(
                        f"[dagger] train batch={batch_idx + 1}: start ({batch_shapes}; "
                        f"{_cuda_status()})",
                        flush=True,
                    )
                is_last = batch_idx + 1 == num_batches
                remainder = num_batches % accumulation
                should_step = (batch_idx + 1) % accumulation == 0 or is_last
                is_remainder_group = (
                    remainder > 0 and batch_idx >= num_batches - remainder
                )
                group_size = remainder if is_remainder_group else accumulation

                # Match TrainGeneratorWorkspace exactly: prepared-model forward
                # and backward share one sync/no-sync context. Accelerate's
                # prepared forward already applies the configured autocast.
                sync_context = (
                    nullcontext()
                    if should_step
                    else self.accelerator.no_sync(model)
                )
                if trace_batch:
                    print(
                        f"[dagger] train batch={batch_idx + 1}: forward/backward start "
                        f"(sync={should_step}, group={group_size}; {_cuda_status()})",
                        flush=True,
                    )
                with sync_context:
                    losses = model(batch, training=True, normalized=True)
                    self.accelerator.backward(losses["loss"] / group_size)
                if trace_batch:
                    print(
                        f"[dagger] train batch={batch_idx + 1}: forward/backward complete "
                        f"({_cuda_status()})",
                        flush=True,
                    )

                if should_step:
                    if trace_batch:
                        print(
                            f"[dagger] train batch={batch_idx + 1}: optimizer step start "
                            f"({_cuda_status()})",
                            flush=True,
                        )
                    self.accelerator.clip_grad_norm_(model.parameters(), 0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    if self.ema is not None:
                        self.ema.step(self.accelerator.unwrap_model(model))
                    if trace_batch:
                        print(
                            f"[dagger] train batch={batch_idx + 1}: optimizer step complete "
                            f"({_cuda_status()})",
                            flush=True,
                        )
                loss_values = {
                    key: float(losses[key].detach().item())
                    for key in ("loss", "flow_loss", "hand_loss")
                }
                for key, value in loss_values.items():
                    sums[key] += value
                if trace_batch:
                    print(
                        f"[dagger] train batch={batch_idx + 1}: loss sync complete "
                        f"({_cuda_status()})",
                        flush=True,
                    )
                # Release model outputs before the next CUDA forward pass.
                del losses
                self.global_step += 1
                completed = batch_idx + 1
                if completed == 1 or completed % progress_every == 0 or is_last:
                    elapsed = time.monotonic() - training_started_at
                    batches_per_second = completed / max(elapsed, 1e-6)
                    eta = (num_batches - completed) / max(batches_per_second, 1e-6)
                    train_progress.set_postfix(
                        loss=f"{loss_values['loss']:.6f}",
                        flow=f"{loss_values['flow_loss']:.6f}",
                        hand=f"{loss_values['hand_loss']:.6f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                        eta=f"{eta:.1f}s",
                        refresh=True,
                    )
                if stall_trace_seconds > 0:
                    faulthandler.cancel_dump_traceback_later()
                    faulthandler.dump_traceback_later(stall_trace_seconds, repeat=True)
        if stall_trace_seconds > 0:
            faulthandler.cancel_dump_traceback_later()
        return {
            "train_loss": sums["loss"] / num_batches,
            "train_flow_loss": sums["flow_loss"] / num_batches,
            "train_hand_loss": sums["hand_loss"] / num_batches,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def save(self, tag: str = "latest") -> Path:
        model = self.accelerator.unwrap_model(self.model)
        state_dicts = {
            "model": _copy_cpu(model.state_dict()),
            "optimizer": _copy_cpu(self.optimizer.state_dict()),
            "lr_scheduler": _copy_cpu(self.scheduler.state_dict()),
        }
        if self.ema_model is not None:
            state_dicts["ema_model"] = _copy_cpu(self.ema_model.state_dict())
        dagger_state = {"iteration": self.iteration, "runtime": self.runtime}
        pickles = {
            "global_step": dill.dumps(self.global_step),
            "scheduler_total_steps": self.payload["pickles"]["scheduler_total_steps"],
            "rng_state": dill.dumps(_rng_state()),
            "dagger_state": dill.dumps(dagger_state),
        }
        if self.ema is not None:
            pickles["ema_optimization_step"] = dill.dumps(self.ema.optimization_step)
        payload = {"cfg": self.flow_cfg, "state_dicts": state_dicts, "pickles": pickles}
        path = self.output_dir / "checkpoints" / f"{tag}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary, pickle_module=dill)
        os.replace(temporary, path)
        return path

    def run(self) -> None:
        from sugar_il.common.json_logger import JsonLogger

        with JsonLogger(str(self.output_dir / "logs.json.txt")) as logger:
            for _ in range(self.cfg.training.iterations):
                rollout = self.collect()
                if rollout is None:
                    break
                train = self.train_latest()
                self.iteration += 1
                log = {
                    "iteration": self.iteration,
                    "global_step": self.global_step,
                    **{key: value for key, value in rollout.items() if key != "termination_reasons"},
                    **train,
                }
                for reason, count in rollout["termination_reasons"].items():
                    log[f"termination/{reason}"] = count
                logger.log(log)
                self.accelerator.log(log, step=self.global_step)
                print(
                    f"Iteration {self.iteration}: windows={rollout['windows']}, "
                    f"loss={train['train_loss']:.6f}, timeout={rollout['time_out_rate']:.1%}"
                )
                print("[dagger] checkpoint save latest starting", flush=True)
                self.save()
                print("[dagger] checkpoint save latest complete", flush=True)
                if self.iteration % self.cfg.checkpoint.every == 0:
                    print(
                        f"[dagger] checkpoint save iteration_{self.iteration:06d} starting",
                        flush=True,
                    )
                    self.save(f"iteration_{self.iteration:06d}")
                    print(
                        f"[dagger] checkpoint save iteration_{self.iteration:06d} complete",
                        flush=True,
                    )
        self.accelerator.end_training()


def main() -> None:
    launch_cwd = Path.cwd()
    sonic_root = Path(__file__).resolve().parents[4]
    os.chdir(sonic_root)
    for path in (sonic_root, sonic_root / "sugar_il"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise RuntimeError("Run DAgger training inside the SONIC Isaac Lab environment") from exc

    from omegaconf import OmegaConf

    launcher_args, overrides = _parser(AppLauncher).parse_known_args()
    config_path = launcher_args.config.expanduser()
    if not config_path.is_absolute():
        config_path = (launch_cwd / config_path).resolve()
    cfg = OmegaConf.merge(
        OmegaConf.load(config_path),
        OmegaConf.from_dotlist(overrides),
    )
    if not 0 <= cfg.training.teacher_ratio <= 1:
        raise ValueError("training.teacher_ratio must be in [0, 1]")
    for path in ("training.iterations", "training.epochs_per_rollout", "checkpoint.every"):
        if OmegaConf.select(cfg, path) < 1:
            raise ValueError(f"{path} must be positive")
    if cfg.rollout.horizon != 40 or cfg.rollout.execution_horizon > cfg.rollout.horizon:
        raise ValueError("Flow DAgger requires horizon=40 and execution_horizon <= 40")
    if cfg.rollout.progress_every < 1:
        raise ValueError("rollout.progress_every must be positive")
    if cfg.training.batch_size is not None and cfg.training.batch_size < 1:
        raise ValueError("training.batch_size must be positive")
    if cfg.training.progress_every < 1:
        raise ValueError("training.progress_every must be positive")
    if cfg.environment.max_parallel_envs < 1:
        raise ValueError("environment.max_parallel_envs must be positive")
    if cfg.environment.timeline_end_time <= 0:
        raise ValueError("environment.timeline_end_time must be positive")

    def resolve(path: Path) -> Path:
        path = Path(path).expanduser()
        return path.resolve() if path.is_absolute() else (launch_cwd / path).resolve()

    required = ("input", "teacher_checkpoint", "generator_checkpoint")
    missing = [name for name in required if cfg.paths.get(name) is None]
    if missing:
        raise ValueError("Missing config paths: " + ", ".join(missing))
    for name in (*required, "output_dir"):
        cfg.paths[name] = str(resolve(cfg.paths[name]))
    if cfg.paths.dataset_root is not None:
        cfg.paths.dataset_root = str(resolve(cfg.paths.dataset_root))

    from sugar_il.workspace.dagger_train.environment import (
        build_teacher,
        load_sonic_config,
        prepare_evaluation_env,
        resolve_motion_inputs,
    )

    robot_dir, dataset_root, keys = resolve_motion_inputs(cfg)
    for name in ("teacher_checkpoint", "generator_checkpoint"):
        path = Path(cfg.paths[name])
        if not path.is_file():
            raise FileNotFoundError(f"paths.{name} does not exist: {path}")
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    num_envs = min(len(keys), int(cfg.environment.max_parallel_envs))
    print(
        f"[dagger] motions={len(keys)}, max_parallel_envs="
        f"{cfg.environment.max_parallel_envs}, creating envs={num_envs}",
        flush=True,
    )
    sonic_cfg = load_sonic_config(cfg, robot_dir, dataset_root, keys, num_envs)
    launcher_args.num_envs = num_envs
    launcher_args.seed = cfg.seed
    launcher_args.enable_cameras = False
    launcher_args.multi_gpu = False
    launcher_args.distributed = False
    launcher_args.output_dir = cfg.paths.output_dir
    launcher_args.env_spacing = sonic_cfg.manager_env.config.env_spacing
    # AppLauncher treats enable_cameras=False as a fallback to ENABLE_CAMERAS.
    # DAgger consumes only state/proprioceptive observations, so an inherited
    # camera environment variable must never select the offscreen renderer.
    os.environ["ENABLE_CAMERAS"] = "0"

    accelerator = _create_accelerator(cfg)
    from gear_sonic.utils.common import seeding

    seeding(cfg.seed)
    launcher_args.device = str(accelerator.device)
    if launcher_args.headless:
        kit_args = str(getattr(launcher_args, "kit_args", "") or "").strip()
        kit_args += (
            " --no-window"
            " --/renderer/multiGpu/maxGpuCount=1"
            " --/omni/replicator/asyncRendering=false"
            " --/app/renderer/waitIdle=true"
            " --/app/hydraEngine/waitIdle=true"
            " --/app/execution/debug/forceSerial=true"
        )
        launcher_args.kit_args = kit_args.strip()
    launcher = AppLauncher(launcher_args)
    try:
        # Isaac Sim can inherit a short (often 10 s) timeline end time.  Once
        # it is reached, SimulationContext's STOP handler waits in a render
        # loop while the timeline is paused.  This looks like a hang at 0/N in
        # the subsequent training loop and leaves CUDA utilization at zero.
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()

        def extend_timeline() -> None:
            timeline.set_looping(False)
            timeline.set_end_time(float(cfg.environment.timeline_end_time))

        extend_timeline()
        from gear_sonic import train_agent_trl

        env = train_agent_trl.create_manager_env(sonic_cfg, launcher_args.device, launcher_args)
        _check_render_state(env)
        print("[dagger] manager environment created; loading teacher", flush=True)
        teacher = build_teacher(
            sonic_cfg,
            env,
            launcher_args.device,
            Path(cfg.paths.teacher_checkpoint),
        )
        print("[dagger] teacher loaded", flush=True)
        env.reinit_dr()
        initial_obs = prepare_evaluation_env(env)
        # Stage/environment initialization may replace timeline settings.
        extend_timeline()
        print(
            f"[dagger] timeline configured: end_time="
            f"{float(cfg.environment.timeline_end_time):.1f}s",
            flush=True,
        )
        print("[dagger] loading flow policy and resume state", flush=True)
        startup_trace_seconds = int(cfg.rollout.get("startup_trace_seconds", 90))
        if startup_trace_seconds > 0:
            faulthandler.dump_traceback_later(startup_trace_seconds, repeat=True)
        try:
            trainer = DaggerTrainer(
                cfg,
                env,
                teacher,
                launcher.app,
                initial_obs,
                accelerator,
            )
        finally:
            if startup_trace_seconds > 0:
                faulthandler.cancel_dump_traceback_later()
        print("[dagger] flow policy loaded; starting DAgger", flush=True)
        trainer.run()
        env.env.close()
    finally:
        # A Ctrl+C may leave the rollout stall watchdog armed while Isaac Sim
        # performs its comparatively slow shutdown/render cleanup.
        faulthandler.cancel_dump_traceback_later()
        launcher.app.close()


if __name__ == "__main__":
    main()
