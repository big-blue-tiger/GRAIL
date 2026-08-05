if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
import pickle
import csv
import dill
import time
from contextlib import nullcontext

from sugar_il.common.pytorch_util import dict_apply
from sugar_il.workspace.base_workspace import BaseWorkspace
from sugar_il.policy.generator import Generator
from sugar_il.dataset.base_dataset import BaseLowdimDataset
from sugar_il.env_runner.generator_runner import GeneratorRunner
from sugar_il.common.checkpoint_util import TopKCheckpointManager
from sugar_il.common.json_logger import JsonLogger
from sugar_il.model.flowmatching.ema_model import EMAModel
from sugar_il.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
OmegaConf.register_new_resolver("eval", eval, replace=True)


def wait_for_everyone(accelerator):
    """Barrier that also works with Accelerate MULTI_CPU plus visible CUDA."""
    if accelerator.num_processes > 1 and accelerator.device.type == "cpu":
        torch.distributed.barrier()
    else:
        accelerator.wait_for_everyone()


def save_loss_curve(output_dir, train_history, val_history):
    """Persist epoch losses as both a CSV table and a PNG plot."""
    csv_path = os.path.join(output_dir, "loss_curve.csv")
    train_by_epoch = dict(train_history)
    val_by_epoch = dict(val_history)
    epoch_progresses = sorted(set(train_by_epoch) | set(val_by_epoch))
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("epoch_progress", "train_loss", "val_loss"))
        for epoch_progress in epoch_progresses:
            writer.writerow(
                (
                    epoch_progress,
                    train_by_epoch.get(epoch_progress, ""),
                    val_by_epoch.get(epoch_progress, ""),
                )
            )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    if train_history:
        train_epochs, train_losses = zip(*train_history)
        axis.plot(train_epochs, train_losses, label="train_loss", linewidth=1.5)
    if val_history:
        val_epochs, val_losses = zip(*val_history)
        axis.plot(val_epochs, val_losses, label="val_loss", marker="o", markersize=3)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training and Validation Loss")
    axis.grid(True, alpha=0.3)
    if train_history or val_history:
        axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=160)
    plt.close(figure)


def save_attention_maps(output_dir, split, epoch, attention_maps):
    directory = os.path.join(output_dir, f"{split}_sample_attn_maps")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{epoch}.pkl"), "wb") as file:
        pickle.dump(attention_maps, file)


class TrainGeneratorWorkspace(BaseWorkspace):
    include_keys = [
        "global_step",
        "epoch",
        "train_loss_history",
        "val_loss_history",
        "scheduler_total_steps",
        "rng_state",
        "ema_optimization_step",
    ]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: Generator
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: Generator = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.global_step = 0
        self.epoch = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.scheduler_total_steps = None
        self.rng_state = None
        self.ema_optimization_step = 0
        self._resume_payload = None
        self._resume_lr_override = None

        if cfg.start_ckpt_path is not None:
            if not cfg.training.resume:
                raise ValueError("start_ckpt_path requires training.resume=true")
            print(f"Starting from checkpoint {cfg.start_ckpt_path}")
            payload = torch.load(
                cfg.start_ckpt_path,
                pickle_module=dill,
                map_location="cpu",
                weights_only=False,
            )
            self._validate_resume_config(payload["cfg"])
            self._resume_payload = payload
            self.load_payload(
                payload,
                exclude_keys=("optimizer", "lr_scheduler"),
                include_keys=self.include_keys,
            )

    def _validate_resume_config(self, saved_cfg):
        """Reject silent changes to optimization/model semantics during resume."""
        paths = (
            "shape_meta",
            "policy",
            "optimizer.weight_decay",
            "optimizer.betas",
            "dataloader.batch_size",
            "training.gradient_accumulate_every",
            "training.lr_scheduler",
            "training.lr_warmup_steps",
            "training.scheduler_num_epochs",
            "training.use_ema",
        )
        mismatches = []
        for path in paths:
            current = OmegaConf.select(self.cfg, path)
            saved = OmegaConf.select(saved_cfg, path)
            if OmegaConf.is_config(current):
                current = OmegaConf.to_container(current, resolve=False)
            if OmegaConf.is_config(saved):
                saved = OmegaConf.to_container(saved, resolve=False)
            if current != saved:
                mismatches.append(path)

        current_lr = float(OmegaConf.select(self.cfg, "optimizer.lr"))
        saved_lr = float(OmegaConf.select(saved_cfg, "optimizer.lr"))
        if current_lr != saved_lr:
            # Changing only the peak LR is a supported fine-tuning operation.
            # The optimizer and scheduler states are still restored below, then
            # the saved schedule is rebased to this new peak value.
            self._resume_lr_override = (saved_lr, current_lr)
        if mismatches:
            raise ValueError(
                "Resume would change training semantics: " + ", ".join(mismatches)
            )

    def _apply_resume_lr_override(self):
        """Rebase a restored scheduler while preserving its current position."""
        if self._resume_lr_override is None:
            return

        saved_lr, requested_lr = self._resume_lr_override
        group_count = len(self.optimizer.param_groups)
        self.lr_scheduler.base_lrs = [requested_lr] * group_count
        current_lrs = []
        for index, param_group in enumerate(self.optimizer.param_groups):
            lr_lambda = self.lr_scheduler.lr_lambdas[index]
            current_lr = requested_lr * lr_lambda(self.lr_scheduler.last_epoch)
            param_group["initial_lr"] = requested_lr
            param_group["lr"] = current_lr
            current_lrs.append(current_lr)
        self.lr_scheduler._last_lr = current_lrs

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                "Resumed scheduler with peak learning rate changed from "
                f"{saved_lr:.2e} to {requested_lr:.2e}; "
                f"current learning rate is {current_lrs[0]:.2e}."
            )

    @staticmethod
    def _capture_rng_state():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _restore_rng_state(self):
        if self.rng_state is None:
            return
        random.setstate(self.rng_state["python"])
        np.random.set_state(self.rng_state["numpy"])
        torch.set_rng_state(self.rng_state["torch"])
        if self.rng_state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.rng_state["cuda"])

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # LOCAL_RANK is populated by ``accelerate launch``/``torchrun``.  Do not
        # force CUDA here so that configuration inspection and CPU smoke tests
        # still work on machines without a GPU.
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        use_cpu = str(cfg.training.device).startswith("cpu")
        if torch.cuda.is_available() and not use_cpu:
            torch.cuda.set_device(local_rank)
        mixed_precision = str(cfg.training.mixed_precision)
        if (
            mixed_precision == "bf16"
            and torch.cuda.is_available()
            and not use_cpu
            and not torch.cuda.is_bf16_supported()
        ):
            mixed_precision = "fp16"
            cfg.training.mixed_precision = mixed_precision
            self.cfg.training.mixed_precision = mixed_precision
            if local_rank == 0:
                print(
                    "BF16 is not supported by these GPUs; using FP16 instead.",
                    flush=True,
                )
        project_config = ProjectConfiguration(
            project_dir=self.output_dir,
            logging_dir=os.path.join(self.output_dir,"tb")
        )
        accelerator = Accelerator(
            log_with='tensorboard', # wandb
            project_config=project_config,
            mixed_precision=mixed_precision,
            device_placement=True,
            cpu=use_cpu,
        )
        # Model initialization happened before distributed setup and therefore
        # used the same seed on every rank. From this point onward, give each
        # rank an independent (but reproducible) data/noise RNG stream.
        set_seed(cfg.training.seed, device_specific=True)

        if accelerator.is_main_process:
            print(f"Using mixed precision: {accelerator.mixed_precision}")
            print(f"Using device: {accelerator.device}")
            print(f"Local rank: {local_rank}")
            print(f"Distributed processes: {accelerator.num_processes}")
            print(
                "Flow matching: "
                f"steps={self.model.num_inference_steps}, "
                f"time_buckets={self.model.num_timestep_buckets}"
            )
            if torch.cuda.is_available() and not use_cpu:
                print(f"CUDA Device: {torch.cuda.get_device_name(local_rank)}")
                print(f"CUDA Capability: {torch.cuda.get_device_capability(local_rank)}")

        # wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        # wandb_cfg.pop('project')
        # wandb_cfg['mode'] = "offline"
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            # config=OmegaConf.to_container(cfg, resolve=True),
            # init_kwargs={"wandb": wandb_cfg}
        )

        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            print(
                f"Computing normalizer from {len(dataset):,} training windows...",
                flush=True,
            )
            normalizer_start_time = time.monotonic()
            normalizer = dataset.get_normalizer()
            with open(normalizer_path, 'wb') as file:
                pickle.dump(normalizer, file)
            print(
                "Normalizer ready in "
                f"{time.monotonic() - normalizer_start_time:.1f}s",
                flush=True,
            )

        # load normalizer on all processes
        wait_for_everyone(accelerator)
        with open(normalizer_path, 'rb') as file:
            normalizer = pickle.load(file)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        if self.scheduler_total_steps is None:
            scheduler_epochs = (
                cfg.training.scheduler_num_epochs
                if cfg.training.scheduler_num_epochs is not None
                else cfg.training.num_epochs
            )
            self.scheduler_total_steps = (
                (
                    len(train_dataloader)
                    + cfg.training.gradient_accumulate_every
                    - 1
                )
                // cfg.training.gradient_accumulate_every
            ) * scheduler_epochs

        # configure lr scheduler
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=self.scheduler_total_steps,
            last_epoch=-1,
        )
        if self._resume_payload is not None:
            states = self._resume_payload["state_dicts"]
            required = {"optimizer", "lr_scheduler"}
            missing = required.difference(states)
            if missing:
                raise KeyError(
                    "Full resume checkpoint is missing: " + ", ".join(sorted(missing))
                )
            self.optimizer.load_state_dict(states["optimizer"])
            self.lr_scheduler.load_state_dict(states["lr_scheduler"])
            self._apply_resume_lr_override()

        # configure ema
        self.ema: EMAModel = None
        if cfg.training.use_ema:
            self.ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            self.ema.optimization_step = self.ema_optimization_step

        # configure env
        env_runner: GeneratorRunner = None
        if accelerator.is_main_process:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, GeneratorRunner)

        # configure checkpoint
        topk_manager = None
        if accelerator.is_main_process:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, self.lr_scheduler
        )
        device = accelerator.device
        if self.ema_model is not None:
            self.ema_model.to(device)
        if self._resume_payload is not None:
            self._restore_rng_state()
            if accelerator.num_processes > 1 and not accelerator.is_main_process:
                # Checkpoints contain rank-zero RNG state. Preserve it exactly
                # on rank zero and deterministically decorrelate worker ranks.
                set_seed(
                    cfg.training.seed + self.global_step,
                    device_specific=True,
                )

        # save batch for sampling
        train_sampling_batch = None
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        logger_context = (
            JsonLogger(log_path) if accelerator.is_main_process else nullcontext(None)
        )
        with logger_context as json_logger:
            def run_validation(progress):
                """Run distributed validation and return the global mean loss."""
                if len(val_dataloader) == 0:
                    return None

                was_training = self.model.training
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    with tqdm.tqdm(
                        val_dataloader,
                        desc=(
                            f"Validation epoch {self.epoch} "
                            f"({progress:.0%})"
                        ),
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                        disable=not accelerator.is_main_process,
                    ) as val_tepoch:
                        for val_batch_idx, val_batch in enumerate(val_tepoch):
                            val_batch = dict_apply(
                                val_batch,
                                lambda x: x.to(device, non_blocking=True),
                            )
                            val_loss_dict = self.model(val_batch, training=False)
                            val_losses.append(val_loss_dict["loss"].item())
                            if (
                                cfg.training.max_val_steps is not None
                                and val_batch_idx
                                >= cfg.training.max_val_steps - 1
                            ):
                                break

                val_loss_stats = torch.tensor(
                    [sum(val_losses), len(val_losses)],
                    device=accelerator.device,
                    dtype=torch.float64,
                )
                val_loss_stats = accelerator.reduce(
                    val_loss_stats, reduction="sum"
                )
                if was_training:
                    self.model.train()
                if val_loss_stats[1] <= 0:
                    return None

                val_loss = (val_loss_stats[0] / val_loss_stats[1]).item()
                if accelerator.is_main_process:
                    epoch_progress = self.epoch + progress
                    self.val_loss_history.append((epoch_progress, float(val_loss)))
                    if cfg.training.save_loss_curve:
                        save_loss_curve(
                            self.output_dir,
                            self.train_loss_history,
                            self.val_loss_history,
                        )
                    print(
                        f"Epoch {self.epoch} validation at {progress:.0%}: "
                        f"avg={val_loss:.6f}, "
                        f"batches={int(val_loss_stats[1].item())}"
                    )
                return val_loss

            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()

                train_losses = list()
                previous_train_loss = None
                num_train_batches = len(train_dataloader)
                if cfg.training.max_train_steps is not None:
                    num_train_batches = min(
                        num_train_batches, cfg.training.max_train_steps
                    )
                val_checks_per_epoch = max(
                    1, int(cfg.training.val_checks_per_epoch)
                )
                validation_batches = {
                    max(
                        1,
                        min(
                            num_train_batches,
                            (check_idx * num_train_batches
                             + val_checks_per_epoch - 1)
                            // val_checks_per_epoch,
                        ),
                    )
                    for check_idx in range(1, val_checks_per_epoch + 1)
                }
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec,
                        disable=not accelerator.is_main_process) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        # Sampling is a main-process-only side task. Keeping this
                        # batch only on rank zero also avoids wasting GPU memory.
                        if accelerator.is_main_process:
                            train_sampling_batch = batch

                        is_last_batch = batch_idx == (num_train_batches - 1)
                        remainder = (
                            num_train_batches
                            % cfg.training.gradient_accumulate_every
                        )
                        # step optimizer
                        should_step = (
                            (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0
                            or is_last_batch
                        )
                        is_remainder_group = (
                            remainder > 0
                            and batch_idx >= num_train_batches - remainder
                        )
                        accumulation_size = (
                            remainder
                            if is_remainder_group
                            else cfg.training.gradient_accumulate_every
                        )
                        # Skip redundant gradient all-reduces on accumulation
                        # micro-batches; the final backward synchronizes the sum.
                        sync_context = (
                            nullcontext()
                            if should_step
                            else accelerator.no_sync(self.model)
                        )
                        with sync_context:
                            # DDP requires both forward and backward to be inside
                            # no_sync for an accumulation micro-batch.
                            loss_dict = self.model(batch, training=True)
                            raw_loss = loss_dict['loss']
                            loss = raw_loss / accumulation_size
                            accelerator.backward(loss)

                        if should_step:
                            accelerator.clip_grad_norm_(
                                self.model.parameters(), max_norm=0.5
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.lr_scheduler.step()

                        # EMA tracks optimizer updates, not micro-batches.
                        if cfg.training.use_ema and should_step:
                            self.ema.step(accelerator.unwrap_model(self.model))
                            self.ema_optimization_step = self.ema.optimization_step

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        flow_loss_cpu = loss_dict['flow_loss'].item()
                        hand_loss_cpu = loss_dict['hand_loss'].item()
                        train_losses.append(raw_loss_cpu)
                        loss_delta = (
                            0.0 if previous_train_loss is None
                            else raw_loss_cpu - previous_train_loss
                        )
                        previous_train_loss = raw_loss_cpu
                        tepoch.set_postfix(
                            loss=f"{raw_loss_cpu:.6f}",
                            avg=f"{np.mean(train_losses):.6f}",
                            delta=f"{loss_delta:+.2e}",
                            flow=f"{flow_loss_cpu:.6f}",
                            hand=f"{hand_loss_cpu:.6f}",
                            lr=f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                            refresh=False,
                        )
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'train_flow_loss': flow_loss_cpu,
                            'train_hand_loss': hand_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': self.lr_scheduler.get_last_lr()[0]
                        }

                        completed_batches = batch_idx + 1
                        should_validate = (
                            self.epoch % cfg.training.val_every == 0
                            and completed_batches in validation_batches
                        )
                        if should_validate:
                            val_loss = run_validation(
                                completed_batches / num_train_batches
                            )
                            if val_loss is not None:
                                step_log['val_loss'] = val_loss

                        if not is_last_batch:
                            if accelerator.is_main_process:
                                # The last step is combined with validation/rollout.
                                accelerator.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss_stats = torch.tensor(
                    [sum(train_losses), len(train_losses)],
                    device=accelerator.device,
                    dtype=torch.float64,
                )
                train_loss_stats = accelerator.reduce(train_loss_stats, reduction="sum")
                train_loss = (
                    train_loss_stats[0] / train_loss_stats[1].clamp_min(1)
                ).item()
                step_log['train_loss'] = train_loss
                if accelerator.is_main_process:
                    print(
                        f"Epoch {self.epoch} loss summary: "
                        f"avg={train_loss:.6f}, last={train_losses[-1]:.6f}, "
                        f"flow={step_log['train_flow_loss']:.6f}, "
                        f"hand={step_log['train_hand_loss']:.6f}"
                    )

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 \
                        and accelerator.is_main_process:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)
                # Other ranks must not enter the next DDP forward while rank zero
                # is still performing rollout inference.
                wait_for_everyone(accelerator)

                if accelerator.is_main_process:
                    # Epoch averages are positioned at the end of each epoch so
                    # they share the same x-axis as intra-epoch validation.
                    self.train_loss_history.append((self.epoch + 1.0, float(train_loss)))
                    if cfg.training.save_loss_curve \
                            and (self.epoch % cfg.training.loss_curve_every) == 0:
                        save_loss_curve(
                            self.output_dir,
                            self.train_loss_history,
                            self.val_loss_history,
                        )

                # def log_action_mse(step_log, category, pred_action, gt_action):
                #     B, T, _ = pred_action.shape
                #     pred_action = pred_action.view(B, T, -1)
                #     gt_action = gt_action.view(B, T, -1)
                #     step_log[f'{category}_action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                def log_action_metrics(step_log, category, prediction, target):
                    latent_mse = torch.nn.functional.mse_loss(prediction['latent'], target['latent'])
                    hand = prediction['hand_primitive'].bool()
                    target_hand = target['hand_primitive'].bool()
                    step_log[f'{category}_latent_mse'] = latent_mse.item()
                    step_log[f'{category}_hand_accuracy'] = (hand == target_hand).float().mean().item()
                    step_log[f'{category}_hand_sequence_accuracy'] = (hand == target_hand).all(dim=(-1, -2)).float().mean().item()
                    for hand_idx, name in enumerate(('left', 'right')):
                        predicted_positive = hand[..., hand_idx]
                        actual_positive = target_hand[..., hand_idx]
                        tp = (predicted_positive & actual_positive).sum().float()
                        fp = (predicted_positive & ~actual_positive).sum().float()
                        fn = (~predicted_positive & actual_positive).sum().float()
                        step_log[f'{category}_{name}_f1'] = (2 * tp / (2 * tp + fp + fn).clamp_min(1)).item()
                # Run flow-matching sampling on a training batch.
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        gt_action = batch['action']
                        pred_action = policy.predict_action(batch['obs'], gen_attn_map=cfg.training.gen_attn_map)
                        if cfg.training.gen_attn_map:
                            save_attention_maps(
                                self.output_dir,
                                "train",
                                self.epoch,
                                pred_action["attention_maps"],
                            )
                        log_action_metrics(step_log, 'train', pred_action, gt_action)

                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            gt_action = batch['action']
                            pred_action = policy.predict_action(batch['obs'], gen_attn_map=cfg.training.gen_attn_map)
                            if cfg.training.gen_attn_map:
                                save_attention_maps(
                                    self.output_dir,
                                    "val",
                                    self.epoch,
                                    pred_action["attention_maps"],
                                )
                            log_action_metrics(step_log, 'val', pred_action, gt_action)

                        del batch
                        del gt_action
                        del pred_action
                wait_for_everyone(accelerator)

                completed_epoch = self.epoch
                if accelerator.is_main_process:
                    accelerator.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                self.rng_state = self._capture_rng_state()

                # Save the next epoch/global_step so resume never repeats work.
                if (
                    completed_epoch % cfg.training.checkpoint_every
                ) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    topk_ckpt_path = None
                    if cfg.checkpoint.topk.monitor_key in metric_dict:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp

                # Save model at specific epochs without affecting best model saving
                if (completed_epoch % 100 == 0) and accelerator.is_main_process:
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    save_dir = os.path.join(self.output_dir, 'epoch_checkpoints')
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(
                        path=os.path.join(save_dir, f"epoch={completed_epoch}.ckpt")
                    )
                    self.model = model_ddp

        if accelerator.is_main_process:
            self.wait_for_checkpoint()
        wait_for_everyone(accelerator)
        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainGeneratorWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
