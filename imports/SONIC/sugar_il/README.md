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
  dataset_path='../outputs/ego_view/**/*.object_aware.pkl' \
  num_epochs=500 \
  training.val_every=1 \
  checkpoint.topk.monitor_key=val_loss \
  'checkpoint.topk.format_str="epoch-{epoch:04d}-val_loss-{val_loss:.3f}.ckpt"'
```

## Inference

`GeneratorWrapper.observation_from_world()` accepts current robot/object world
poses. It uses the same `world_pose_to_body()` implementation as training.
`predict_from_world()` returns 40 consecutive frames. It includes 64-D latents,
hand primitives, probabilities, logits, and concatenated 66-D actions.

```bash
 python -u sugar_il/workspace/run_generator_isaaclab.py \
  --generator-checkpoint data/outputs/train_generator_ObjectAwareSONIC/checkpoints/epoch-0421-val_loss-0.012.ckpt \
  --seed 24 \
  --device cuda:0 \
  --robot-position -0.31984177  0.60989204  0.78332764 \
  --object-initial-position 0.06525806 0.13028011 0.78545775 \
  --table-position 0.         0.         0.64405456

--robot-position X Y Z
--robot-quaternion-wxyz W X Y Z

--object-initial-position X Y Z
--object-initial-quaternion-wxyz W X Y Z

--table-position X Y Z
--table-quaternion-wxyz W X Y Z

  the robot pose is [-0.41984177  0.69989204  0.78332764], [ 0.70745337 -0.02033162 -0.00723898 -0.70643055]
 the object pose is [0.02525806 0.03028011 0.88545775], [0.7192729  0.68741876 0.07175218 0.07037108]
 the table pose is [0.         0.         0.74405456], [0. 0. 0. 1.]
```
