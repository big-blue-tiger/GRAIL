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
from accelerate.utils import ProjectConfiguration
OmegaConf.register_new_resolver("eval", eval, replace=True)


def save_loss_curve(output_dir, train_history, val_history):
    """Persist epoch losses as both a CSV table and a PNG plot."""
    csv_path = os.path.join(output_dir, "loss_curve.csv")
    val_by_epoch = dict(val_history)
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("epoch", "train_loss", "val_loss"))
        for epoch, train_loss in train_history:
            writer.writerow((epoch, train_loss, val_by_epoch.get(epoch, "")))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    train_epochs, train_losses = zip(*train_history)
    axis.plot(train_epochs, train_losses, label="train_loss", linewidth=1.5)
    if val_history:
        val_epochs, val_losses = zip(*val_history)
        axis.plot(val_epochs, val_losses, label="val_loss", marker="o", markersize=3)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training and Validation Loss")
    axis.grid(True, alpha=0.3)
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
            "optimizer",
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
        if mismatches:
            raise ValueError(
                "Resume would change training semantics: " + ", ".join(mismatches)
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

        # Set GPU device before initializing accelerator
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        project_config = ProjectConfiguration(
            project_dir=self.output_dir,
            logging_dir=os.path.join(self.output_dir,"tb")
        )
        accelerator = Accelerator(
            log_with='tensorboard', # wandb
            project_config=project_config,
            mixed_precision='bf16',  # Enable BF16 mixed precision training
            device_placement=True
        )

        if accelerator.is_main_process:
            print(f"Using mixed precision: {accelerator.mixed_precision}")
            print(f"Using device: {accelerator.device}")
            print(f"Local rank: {local_rank}")
            print(
                "Flow matching: "
                f"steps={self.model.num_inference_steps}, "
                f"time_buckets={self.model.num_timestep_buckets}"
            )
            if torch.cuda.is_available():
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
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

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

        # configure ema
        self.ema: EMAModel = None
        if cfg.training.use_ema:
            self.ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            self.ema.optimization_step = self.ema_optimization_step

        # configure env
        env_runner: GeneratorRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, GeneratorRunner)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, self.lr_scheduler
        )
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)
        if self._resume_payload is not None:
            self._restore_rng_state()

        # save batch for sampling
        train_sampling_batch = None
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
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
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec,
                        disable=not accelerator.is_main_process) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        loss_dict = self.model.compute_loss(batch, training=True)
                        raw_loss = loss_dict['loss']
                        is_last_batch = batch_idx == (num_train_batches - 1)
                        remainder = (
                            num_train_batches
                            % cfg.training.gradient_accumulate_every
                        )
                        accumulation_size = (
                            remainder
                            if is_last_batch and remainder
                            else cfg.training.gradient_accumulate_every
                        )
                        loss = raw_loss / accumulation_size
                        accelerator.backward(loss)

                        # step optimizer
                        should_step = (
                            (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0
                            or is_last_batch
                        )
                        if should_step:
                            torch.nn.utils.clip_grad_norm_(
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

                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
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
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec,
                                disable=not accelerator.is_main_process) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss_dict = self.model.compute_loss(batch, training=False)
                                val_losses.append(loss_dict['loss'])
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break

                        if len(val_losses) > 0:
                            # Collect validation losses from all processes
                            val_losses = torch.stack(val_losses)
                            val_losses = accelerator.gather(val_losses)

                            # Calculate mean loss on main process
                            if accelerator.is_main_process:
                                val_loss = torch.mean(val_losses).item()
                                step_log['val_loss'] = val_loss
                                print(
                                    f"Epoch {self.epoch} validation summary: "
                                    f"avg={val_loss:.6f}, batches={len(val_losses)}"
                                )

                if accelerator.is_main_process:
                    self.train_loss_history.append((self.epoch, float(train_loss)))
                    if 'val_loss' in step_log:
                        self.val_loss_history.append((self.epoch, float(step_log['val_loss'])))
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

                completed_epoch = self.epoch
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

        self.wait_for_checkpoint()
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
