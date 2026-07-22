# SONIC Object-Aware DiT

This package trains a 16-frame diffusion policy directly from trusted SONIC
`schema_version=3` `*.object_aware.pkl` recordings. Current and goal object
poses are recomputed from world-frame poses in the current robot root frame;
the legacy reference `object_pos_b`, `object_ori_b_6d`, and
`target_object_pos` arrays are intentionally ignored.

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

The dataset uses frame `t-1` as the last executed action and predicts the real,
uninterpolated frames `t:t+16`. A 499-frame recording yields 483 samples.

## Inference

`GeneratorWrapper.observation_from_world()` accepts current robot/object world
poses and the fixed goal world pose. It uses the same `world_pose_to_body()`
implementation as training and returns the nine model observation tensors.
`predict_from_world()` returns 16 full 64-D latents, binary hand primitives,
probabilities, logits, and their concatenated 66-D actions.

```bash
 python -u sugar_il/workspace/run_generator_isaaclab.py \
  --generator-checkpoint data/outputs/2026.07.22/15.49_train_generator_ObjectAwareSONIC/checkpoints/epoch-0018-val_loss-0.018.ckpt \
  --seed 24 \
  --device cuda:0
```