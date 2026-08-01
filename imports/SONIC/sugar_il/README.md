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
  --generator-checkpoint sugar_il/data/outputs/train_generator_ObjectAwareSONIC/checkpoints/epoch-0421-val_loss-0.012.ckpt \
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
  --checkpoint logs_rl/pnp_table_pnp_table-721/last.pt \
  --generator-checkpoint sugar_il/data/outputs/dagger_automation/2026.07.30-00.36.52-978188/round_10/checkpoints/latest.ckpt \
  --input ../../../sbto/datas/sbto_to_grail/pickup_table/robot \
  --no-video-rendering \
  --output-dir outputs/dagger
  --batch-size 128
```

#### use bash
```bash
cd /home/tide/robot/GRAIL/imports/SONIC/sugar_il
DAGGER_ROUNDS=10 GPU=0 ./run_dagger.sh
```
