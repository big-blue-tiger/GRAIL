# Task General Tracking

Task-general tracking trains physics-based policies on retargeted GRAIL motion
libraries: pick-and-place, manipulation, terrain-aware tracking, and locomotion.
The training implementation lives in the
{src}`imports/SONIC <imports/SONIC/>`
vendored release tree.

This page documents the GRAIL-specific pieces:

- The four self-contained HOI release configs (`pnp_*`, `advanced_manip_*`)
  used to train pick-and-place and manipulation policies.
- The shared `terrain_tracking` config used for scene/terrain-aware tracking.

For lower-level training internals see
{blob}`imports/SONIC/README.md`
and
{src}`imports/SONIC/docs/`.

## Install

Same `sonic` env as [retargeting](retargeting.md#install):

```bash
bash scripts/setup/install_env_sonic.sh
```

This sets up IsaacLab, GMR, GRAIL, and the training stack in one conda env.

## Checkpoints

The base behavior-model warm-start and the task-specific reference
bundles for finetuning are all fetched by the project-wide setup script:

```bash
bash scripts/setup/download_checkpoints.sh           # all submodules
bash scripts/setup/download_checkpoints.sh \
    --skip-gem-smpl --skip-gem-soma \
    --skip-foundationpose --skip-hunyuan3d           # SONIC only
```

This lands them under `imports/SONIC/models/`:

```
imports/SONIC/models/
├── sonic_manipulation_base/   # base warm-start for pickup and manipulation: last.pt + model_config.yaml
├── pnp_table/                 # pickup table reference: last.pt + config.yaml
├── pnp_ground/                # pickup ground reference: last.pt + config.yaml
└── terrain_stairs/            # terrain stairs reference: last.pt + config.yaml
```

Store path references in experiment configs as relative to
`imports/SONIC/`, not to the GRAIL root — e.g. `models/pnp_table/last.pt`.
The training commands below assume `cd imports/SONIC` first, so those
relative paths resolve correctly.

## Preparing retargeted data for training

1. Retarget as in [retargeting.md](retargeting.md). The GRAIL retargeting
   pipeline writes a `<name>_ha/` directory containing `robot/`, `objects/`,
   `object_usd/`, and `meta/`.
2. Move the retargeted folder to `data/motion_lib/<name>_ha/` — that
   prefix is what the training config loaders expect.
3. For multi-object HOI sweeps, also place the BPS encodings under
   `data/motion_lib/<name>/bps/` (BPS is data-only; it is not
   parameterized by the hand-action variant).
4. For terrain-aware data, re-retarget with `--zero_out_wrist` to skip
   hand IK — see {ref}`terrain-sitting-data`.

## Task-general tracking configs overview

Current state of `imports/SONIC/gear_sonic/config/exp/manager/universal_token/`:

| Path                             | Purpose                                                                                          |
|----------------------------------|--------------------------------------------------------------------------------------------------|
| `scene/terrain_tracking.yaml`    | Shared height-map + object-state terrain-aware tracking config used by the `tnfh`, `tnfhp1`, and `tnch` wrappers |
| `hoi/pnp_table.yaml`             | Self-contained tabletop pick-up release config                                                   |
| `hoi/pnp_ground.yaml`            | Self-contained ground pick-up release config                                                     |
| `hoi/advanced_manip_table.yaml`  | Self-contained advanced-manipulation tabletop config                                             |
| `hoi/advanced_manip_ground.yaml` | Self-contained advanced-manipulation ground config                                               |

The release configs share the same launch pattern: choose the Hydra config,
then pass runtime data paths through Hydra overrides.

## Quick smoke test

A single-GPU, 4-env, 3-iteration run against the `pnp_table` release config —
enough to verify the install end-to-end. Completes in ~2 minutes on a single
L40. Set `DATA_DIR` and `BPS_DIR` to a retargeted motion library prepared as
described above.
### REPLAY
```bash
python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/hoi/pnp_table \
    num_envs=1 headless=False \
    ++replay=True \
    ++manager_env.config.render_results=False \
    ++manager_env.config.gpu_collision_stack_size_exp=28 \
    ++manager_env.commands.motion.motion_lib_cfg.target_fps=50 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=../../data/hf_dataset/data/pickup_table/update_robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=../../data/hf_dataset/data/pickup_table/objects \
    ++manager_env.config.object_usd_path=../../data/hf_dataset/data/pickup_table/object_usd \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=../../data/hf_dataset/data/pickup_table/bps 

python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/hoi/pnp_table \
    num_envs=15 headless=False \
    ++replay=True \
    ++manager_env.config.render_results=False \
    ++manager_env.config.gpu_collision_stack_size_exp=28 \
    ++manager_env.commands.motion.motion_lib_cfg.target_fps=50 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=../../../sbto/datas/sbto_to_grail/pickup_table/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=../../../sbto/datas/sbto_to_grail/pickup_table/objects \
    ++manager_env.config.object_usd_path=../../../sbto/datas/sbto_to_grail/pickup_table/object_usd \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=../../../sbto/datas/sbto_to_grail/pickup_table/bps

```
### View
```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline
cd imports/SONIC
python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/hoi/pnp_table \
    num_envs=3 headless=False \
      ++algo.config.init_noise_std=0.4 \
      ++algo.config.entropy_coef=0.03 \
    ++algo.config.num_learning_iterations=90000 \
    ++manager_env.config.gpu_collision_stack_size_exp=27 \
     ++manager_env.commands.motion.motion_lib_cfg.target_fps=50 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=../../../sbto/datas/sbto_to_grail/pickup_table/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=../../../sbto/datas/sbto_to_grail/pickup_table/objects \
    ++manager_env.config.object_usd_path=../../../sbto/datas/sbto_to_grail/pickup_table/object_usd \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=../../../sbto/datas/sbto_to_grail/pickup_table/bps \
      ++resume=True \
    ++checkpoint=models/pnp_table_warmstart/last.pt 
```
bash grail/visualization/scripts/visualize_single.sh data/hf_dataset/data/pickup_table/robot/pickup_table__beer_19__003.pkl
### train
```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline
cd imports/SONIC
CUDA_VISIBLE_DEVICES=0 python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/hoi/pnp_table \
    num_envs=4096 headless=True \
      ++algo.config.init_noise_std=0.4 \
      ++algo.config.entropy_coef=0.03 \
    ++algo.config.num_learning_iterations=90000 \
    ++manager_env.config.gpu_collision_stack_size_exp=30 \
     ++manager_env.commands.motion.motion_lib_cfg.target_fps=50 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=../../data/raw_grail_chaged_obj/pickup_table/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=../../data/raw_grail_chaged_obj/pickup_table/objects \
    ++manager_env.config.object_usd_path=../../data/raw_grail_chaged_obj/pickup_table/object_usd \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=../../data/raw_grail_chaged_obj/pickup_table/bps 


CUDA_VISIBLE_DEVICES=0 python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/hoi/pnp_table \
    num_envs=4096 headless=True \
    ++algo.config.num_learning_iterations=90000 \
    ++manager_env.config.gpu_collision_stack_size_exp=30 \
     ++manager_env.commands.motion.motion_lib_cfg.target_fps=50 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=../../../sbto/datas/sbto_to_grail/pickup_table/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=../../../sbto/datas/sbto_to_grail/pickup_table/objects \
    ++manager_env.config.object_usd_path=../../../sbto/datas/sbto_to_grail/pickup_table/object_usd \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=../../../sbto/datas/sbto_to_grail/pickup_table/bps \
      ++resume=True \
    ++checkpoint=models/pnp_table_warmstart/last.pt
```

```bash
python -u gear_sonic/eval_agent_trl.py \
    +checkpoint=logs_rl/GRAB_Tracking/manager/universal_token/hoi/pnp_table_pnp_table-sbto_data_717/last.pt \
    +num_envs=12 \
    +headless=False \
    ++manager_env.config.gpu_collision_stack_size_exp=28 \
    ++run_eval_loop=True \
    ++manager_env.commands.motion.debug_vis=False \
    ++max_render_steps=30000


    python -u gear_sonic/eval_agent_trl.py \
  +checkpoint=logs_rl/GRAB_Tracking/manager/universal_token/hoi/pnp_table_pnp_table-sbto_data_717/last.pt \
  +num_envs=12 \
  +headless=False \
  ++run_eval_loop=True \
  ++manager_env.commands.motion.debug_vis=True \
  ++max_render_steps=30000 \
  ++manager_env.config.gpu_collision_stack_size_exp=28
```

### render ego video
```bash

python -u gear_sonic/scripts/get_rl_motion_data.py \
  --gpu 1 \
  --checkpoint logs_rl/pnp_table_pnp_table-721/last.pt \
  --input ../../../sbto/datas/sbto_to_grail/pickup_table/robot \
  --output-dir outputs/grail/all \
  --batch-size 16
  --delete-failed-reference-data \
   --batch-size 128 \


while true; do
    python -u gear_sonic/scripts/get_rl_motion_data.py \
        --gpu 1 \
        --checkpoint logs_rl/pnp_table_pnp_table-721/last.pt \
        --input ../../../sbto/datas/sbto_to_grail/pickup_table/robot \
        --output-dir outputs/grail/append 
    code=$?
    echo "程序结束，退出码：${code}；5 秒后重新运行"
    sleep 5
done
  
```
## Running training

### Pick-up and advanced manipulation

All four release configs share a single launch shape — only the Hydra config
name changes. Set `DATA_DIR` to your retargeted motion library (with
`robot/`, `objects/`, `object_usd/` subdirs) and `BPS_DIR` to the matching
BPS encodings; see [Preparing retargeted data for training](#preparing-retargeted-data-for-training)
for the expected layout. The example script below will launch training with
a single node using 8 GPUs.

```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

cd imports/SONIC
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=${HYDRA_CONFIG} \
    num_envs=2048 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects \
    ++manager_env.config.object_usd_path=${DATA_DIR}/object_usd \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=${BPS_DIR}
```

Available release configs:

| Sweep                   | `HYDRA_CONFIG`                                       |
|-------------------------|------------------------------------------------------|
| `pnp_table`             | `manager/universal_token/hoi/pnp_table`              |
| `pnp_ground`            | `manager/universal_token/hoi/pnp_ground`             |
| `advanced_manip_table`  | `manager/universal_token/hoi/advanced_manip_table`   |
| `advanced_manip_ground` | `manager/universal_token/hoi/advanced_manip_ground`  |

Pick-up and manipulation launch inputs:

| Flag | Effect |
|------|--------|
| `+exp=${HYDRA_CONFIG}` | Selects one release config from the table above. |
| `++manager_env.commands.motion.motion_lib_cfg.motion_file=<path>` | Robot-motion directory, usually `${DATA_DIR}/robot`. |
| `++manager_env.commands.motion.motion_lib_cfg.object_motion_file=<path>` | Object-motion directory, usually `${DATA_DIR}/objects`. |
| `++manager_env.config.object_usd_path=<path>` | Object USD directory, usually `${DATA_DIR}/object_usd`. |
| `++manager_env.commands.motion.motion_lib_cfg.bps_dir=<path>` | BPS encoding directory for multi-object pick-up / manipulation data. |

#### Finetuning a pick-up policy

To continue from an existing pick-up run, use the matching reference
bundle (downloaded by `download_checkpoints.sh` above):

| Config | Reference config | Reference checkpoint |
|--------|------------------|----------------------|
| `manager/universal_token/hoi/pnp_table`  | `models/pnp_table/config.yaml`  | `models/pnp_table/last.pt`  |
| `manager/universal_token/hoi/pnp_ground` | `models/pnp_ground/config.yaml` | `models/pnp_ground/last.pt` |

Use the reference `config.yaml` for provenance and evaluation; the finetune
command below selects the public release config and warm-resumes from
`last.pt`. Write the new run to a separate `experiment_dir`.

```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

cd imports/SONIC
python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/hoi/pnp_table \
    num_envs=2048 headless=True \
    ++resume=True \
    ++checkpoint=models/pnp_table/last.pt \
    experiment_dir=${FINETUNE_DIR} \
    ++algo.config.num_learning_iterations=10000 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects \
    ++manager_env.config.object_usd_path=${DATA_DIR}/object_usd \
    ++manager_env.commands.motion.motion_lib_cfg.bps_dir=${BPS_DIR}
```

Use `+exp=manager/universal_token/hoi/pnp_ground` and
`++checkpoint=models/pnp_ground/last.pt` for ground pick-up data.

### Terrain-aware tracking

The current GRAIL terrain-aware runs use one shared height-map + object-state
config. Set `DATA_DIR` to a retargeted dataset root with `robot/`, `objects/`,
and `object_usd/`.

```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

cd imports/SONIC
python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/scene/terrain_tracking \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects \
    ++manager_env.config.terrain_motion_dir=${DATA_DIR}
```

Use the same config for all scene/terrain-aware datasets and pass only the
runtime data paths through Hydra overrides. If the active dataset root does
not include `flat_placeholder.usd`, set
`++manager_env.config.flat_usd_path=<path>` to a compatible placeholder USD.

Terrain launch inputs:

| Flag | Effect |
|------|--------|
| `++manager_env.config.terrain_motion_dir=<path>` | Dataset root with paired `robot/*.pkl` + `object_usd/*.usd` (1:1 stem matching). Auto-discovers all pairs. |
| `++manager_env.config.flat_motion_dir=<path>` | Optional — adds flat (non-terrain) motions, interleaved between terrain envs. |
| `++manager_env.config.flat_usd_path=<path>` | Explicit placeholder USD for non-terrain envs. Falls back to `<terrain_motion_dir>/flat_placeholder.usd` (new layout) or `<terrain_motion_dir>/object_usd/flat_placeholder.usd` (legacy). |
| `++manager_env.config.flat_to_terrain_ratio=R` | Every `(R+1)`th env is terrain; rest are flat. `R=0` → all envs terrain. |

The terrain path emits `/tmp/rank_<R>_motion_keys.txt` per GPU and logs
`[TerrainAutoDiscover]` / `[PerRankUSD]` / `[PerRankMotion]` during init —
grep those to confirm the slicer is doing what you expect.

#### Finetuning a terrain policy

Terrain finetuning uses the same warm-resume pattern: keep
`models/terrain_stairs/config.yaml` with `models/terrain_stairs/last.pt`,
write to a new output directory, and point `DATA_DIR` at the next
terrain-aware motion-library partition.

The published reference bundle (`models/terrain_stairs/`) is the
**stairs** policy only — use it as the warm-start when finetuning on any
stairs-like dataset. Reference bundles for the other terrain types
(curb, slope, etc.) will be published in a future release.

```bash
conda activate sonic
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

cd imports/SONIC
python -u gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/scene/terrain_tracking \
    num_envs=4096 headless=True \
    ++resume=True \
    ++checkpoint=models/terrain_stairs/last.pt \
    experiment_dir=${FINETUNE_DIR} \
    ++algo.config.num_learning_iterations=20000 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/robot \
    ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=${DATA_DIR}/objects \
    ++manager_env.config.terrain_motion_dir=${DATA_DIR}
```

If the dataset root does not provide `flat_placeholder.usd`, also pass
`++manager_env.config.flat_usd_path=<path-to-placeholder-usd>`.

### Multi-node `accelerate` template

Replace the single-node launcher with the multi-node form. Same
`gear_sonic/train_agent_trl.py` command and `${ARGS[@]}`, just different launcher
flags. Example: 8 nodes × 8 GPUs = 64 GPUs.

```bash
accelerate launch \
    --multi_gpu \
    --num_machines=8 \
    --num_processes=64 \
    --machine_rank=$MACHINE_RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    gear_sonic/train_agent_trl.py "${ARGS[@]}" num_envs=2048
```

See the
[Accelerate distributed training guide](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
and
[multi-node launcher docs](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch).

## Output layout

Each run writes to:

```
logs_rl/TRL_G1_Track/manager/<config_path>/<exp_name>-<timestamp>/
├── config.yaml              # full resolved Hydra config
├── model_step_NNNNNN.pt     # checkpoint every N iters (algo.config.save_every)
├── last.pt                  # symlink to the latest step
├── meta.yaml                # wandb_id + misc provenance
└── events.out.tfevents.*    # tensorboard (optional; wandb is primary)
```

W&B run name and project come from the `+opt=wandb` Hydra opt group
(`gear_sonic/config/opt/wandb.yaml`).

## Evaluation

See {src}`imports/SONIC docs <imports/SONIC/docs/>`
for eval workflows. GRAIL does not add eval tooling — eval runs against the
training checkpoint directory format directly.

## Troubleshooting

| Symptom                                         | Cause / fix                                                                                    |
|-------------------------------------------------|------------------------------------------------------------------------------------------------|
| `reward_grasp: gate_with_contact_label=True but no contact label found` | Retarget output is missing `contact_points_{left,right}_hand` — re-run `process.sh` with `--include_contact_points` (it is the default). |
| Wrong motion-lib format                         | Verify `robot/` contains per-motion pkls with keys `joint_pos`, `hand_action_{left,right}`, `table_pos`. |
