# SONIC Object-Aware Flow Matching

This package trains a 40-frame flow-matching policy directly from trusted SONIC
`schema_version=3` `*.object_aware.pkl` recordings. The current object pose is
recomputed from world-frame poses in the current robot root frame. The current
object token also includes the magnitudes of the eight configured hand-object
contact-link forces. The scene-geometry token concatenates the 10-D object BPS
with `table_geometry=[center_height_z, thickness, length_x, width_y]`.

## Train

From this directory, install the package in the SONIC environment and run:

```bash
python -m pip install -e .
python sugar_il/workspace/train_generator_workspace.py \
  task=ObjectAware \
  dataset_path='../outputs/ego_view/**/*.object_aware.pkl' \
  num_epochs=1000

python sugar_il/workspace/train_generator_workspace.py \
  task=ObjectAware \
  dataset_path='../outputs/ego_view/**/*.object_aware.pkl' \
  num_epochs=1000 \
  training.val_every=5 \
  checkpoint.topk.monitor_key=val_loss \
  'checkpoint.topk.format_str="epoch-{epoch:04d}-val_loss-{val_loss:.3f}.ckpt"'
```

The dataset predicts 40 consecutive 50 Hz frames, `t:t+40`. During inference,
the Isaac Lab loop executes the first 20 predicted frames before replanning.

```bash
python sugar_il/workspace/train_generator_workspace.py \
  task=ObjectAware \
  dataset_path='../outputs/**/**/*.object_aware.pkl' \
  num_epochs=500 \
  training.val_every=1 \
  checkpoint.topk.monitor_key=val_loss \
  'checkpoint.topk.format_str="epoch-{epoch:04d}-val_loss-{val_loss:.3f}.ckpt"'
```

To train one model with Distributed Data Parallel on GPUs 0, 1, and 2, launch
one process per GPU. `dataloader.batch_size` is per GPU, so the default effective
batch size is `128 * 3 * training.gradient_accumulate_every`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes 3 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  sugar_il/workspace/train_generator_workspace.py \
  task=ObjectAware \
  dataset_path='../outputs/**/**/*.object_aware.pkl' \
  num_epochs=500 \
  training.val_every=1 \
  checkpoint.topk.monitor_key=val_loss \
  'checkpoint.topk.format_str="epoch-{epoch:04d}-val_loss-{val_loss:.3f}.ckpt"'


CUDA_VISIBLE_DEVICES=0,1,2 \
python -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes 3 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  sugar_il/workspace/train_generator_workspace.py \
  task=ObjectAware \
  dataset_path='../outputs/**/**/*.object_aware.pkl' \
  num_epochs=300 \
  lr=3e-5 \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64 \
  training.val_every=1 \
  checkpoint.topk.monitor_key=val_loss \
  'checkpoint.topk.format_str="epoch-{epoch:04d}-val_loss-{val_loss:.3f}.ckpt"' \
  training.resume=true \
  start_ckpt_path=/home/GRAIL/imports/SONIC/sugar_il/data/outputs/2026.08.02/16.10_train_generator_ObjectAwareSONIC/checkpoints/epoch-0001-val_loss-0.050.ckpt
```

Only rank zero runs rollouts, writes logs, and saves the single resulting model
checkpoint. Training and validation data are sharded automatically across the
three GPUs.

## Inference

`GeneratorWrapper.observation_from_world()` receives the current robot/object
world poses and uses the same `world_pose_to_body()` conversion as training.
Each inference call predicts 40 consecutive frames (64-D latent + 2-D hand
primitive); the simulator executes the first 20 frames and then replans.

The reference `--motion-file` is used for the object frame-zero pose, table
metadata, object USD, and BPS. Reference robot actions are not replayed. The
default path is:

```text
/home/tide/robot/sbto/datas/sbto_to_grail/pickup_table/robot/pickup_table__apple_18__004.pkl
```

```bash
python -u sugar_il/sugar_il/workspace/run_generator_isaaclab.py \
  --generator-checkpoint sugar_il/data/outputs/2026.08.02/17.35_train_generator_ObjectAwareSONIC/checkpoints/epoch-0005-val_loss-0.063.ckpt\
  --mode single --seed 24 --device cuda:0 --no-render-video --headless
```

Single-episode validation uses the reference object/table pose and the fixed
robot initial pose declared near the top of `run_generator_isaaclab.py`:

```bash
python -u sugar_il/sugar_il/workspace/run_generator_isaaclab.py \
  --generator-checkpoint sugar_il/data/outputs/dagger_automation/2026.07.30-15.57.39-31304/round_09/checkpoints/epoch-0382-val_loss-0.014.ckpt \
  --mode single --render-video --headless
```

Batch success-rate test (random table height / object / robot pose per episode):

```bash
python -u sugar_il/sugar_il/workspace/run_generator_isaaclab.py \
  --generator-checkpoint sugar_il/data/outputs/dagger_automation/2026.07.30-15.57.39-31304/round_09/checkpoints/epoch-0382-val_loss-0.014.ckpt \
  --mode batch --episodes 20 --render-video --headless
```

Batch ranges are defined in `RandomizationRanges` near the top of
`run_generator_isaaclab.py`, or overridden with `--table-height-range`,
`--object-x-offset-range`, `--object-y-offset-range`, and robot offset range
arguments. An object lift of `0.10` m by default (configured with
`--lift-height`) counts as a successful grasp. Results are written to
`output/generator_eval/results.json`;
videos, when enabled, are written below `output/generator_eval/videos/`.


## DAGGER GATA GEN
```bash
cd /home/tide/robot/GRAIL/imports/SONIC

python -u sugar_il/sugar_il/workspace/get_generator_data_for_dagger.py \
  --gpu 0 \
  --checkpoint ../models/pnp_table/last.pt \
  --generator-checkpoint data/outputs/2026.08.02/17.35_train_generator_ObjectAwareSONIC/checkpoints/epoch-0005-val_loss-0.063.ckpt  \
  --input ../../../sbto/datas/sbto_to_grail/pickup_table/robot \
  --no-video-rendering \
  --output-dir outputs/dagger
  --batch-size 16

cd /home/tide/robot/GRAIL/imports/SONIC/sugar_il

python -m sugar_il.workspace.dagger_train.train \
  paths.input=../../../../sbto/datas/sbto_to_grail/pickup_table/robot \
  paths.teacher_checkpoint=../models/pnp_table/last.pt  \
  paths.generator_checkpoint=data/outputs/2026.08.02/17.35_train_generator_ObjectAwareSONIC/checkpoints/epoch-0005-val_loss-0.063.ckpt \
  paths.output_dir=data/outputs/dagger_online \
    --headless
  training.iterations=488 \
  training.epochs_per_rollout=1 \
  training.batch_size=64 \
  training.teacher_ratio=0.5 \
  training.mixed_precision=fp16 \
  environment.max_parallel_envs=64 \
  checkpoint.every=10 \
  --headless
```

#### use bash
```bash
cd /home/tide/robot/GRAIL/imports/SONIC/sugar_il
DAGGER_ROUNDS=10 GPU=0 ./run_dagger.sh
```

#### use python
```bash
cd /home/tide/robot/GRAIL/imports/SONIC/sugar_il
python -m sugar_il.workspace.dagger_train.train \
  paths.input=../../../data/hf_dataset/data_update/data/pickup_table/robot \
  paths.teacher_checkpoint=../models/pnp_table/last.pt \
  paths.generator_checkpoint=data/outputs/2026.08.02/17.35_train_generator_ObjectAwareSONIC/checkpoints/epoch-0002-val_loss-0.054.ckpt \
  paths.output_dir=data/outputs/dagger_online_envs2048 \
  training.teacher_ratio=0.5 \
  training.learning_rate=2e-4 \
  --headless \
  environment.max_parallel_envs=2048 \
  training.batch_size=256 


python -m sugar_il.workspace.dagger_train.train \
  paths.input=../../../data/hf_dataset/data/pickup_table_update/robot \
  paths.teacher_checkpoint=../models/pnp_table/last.pt \
  paths.generator_checkpoint=data/outputs/2026.08.02/17.35_train_generator_ObjectAwareSONIC/checkpoints/epoch-0005-val_loss-0.063.ckpt\
  paths.output_dir=data/outputs/dagger_online \
  training.teacher_ratio=1 \
  --headless


```

## clear the dataset
```bash

GRAIL/data/hf_dataset/data_update/data/pickup_table/robot

python -u gear_sonic/scripts/get_rl_motion_data.py \
  --gpu 0 \
  --checkpoint models/pnp_table/last.pt \
  --input ../../data/hf_dataset/data_update/data/pickup_table/robot \
  --output-dir outputs/grail/all \
  --batch-size 8 


  --delete-failed-reference-data \
  --delete-reference-penetration-frames \
  ++manager_env.config.render_results=False \
  ++manager_env.recorders.render_envs=null
```

## TEST generator
```bash
python -u sugar_il/workspace/test_generator_isaaclab.py \
  --generator-checkpoint data/outputs/2026.08.02/17.35_train_generator_ObjectAwareSONIC/checkpoints/epoch-0002-val_loss-0.054.ckpt \
  --episodes 16 \
  --parallel-envs 16 \
  --render-video \
  --headless \
  --device cuda:0
```