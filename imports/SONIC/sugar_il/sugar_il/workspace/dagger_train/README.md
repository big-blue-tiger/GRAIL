# Online DAgger

All DAgger-specific parameters live in `config.yaml`. The optimizer, learning-rate
scheduler, model, normalizer, gradient accumulation, and EMA settings are restored
from the flow checkpoint so resume cannot silently change its training semantics.

Set the three required paths in a copied YAML config and run:

```bash
python -m sugar_il.workspace.dagger_train.train \
  --config sugar_il/sugar_il/workspace/dagger_train/config.yaml \
  paths.input=/home/tide/robot/sbto/datas/sbto_to_grail/pickup_table/robot \
  paths.teacher_checkpoint=../logs_rl/pnp_table_pnp_table-721/last.pt \
  paths.generator_checkpoint=data/outputs/dagger_automation/<run>/<round>/checkpoints/latest.ckpt \
  paths.output_dir=data/outputs/dagger_online \
  --headless
```

Replace `<run>/<round>` with the flow checkpoint to resume. Relative paths are
resolved from the directory where the command is launched.

Any YAML key can be overridden with the same dotted `key=value` form, for example:

```bash
training.teacher_ratio=0.25 training.iterations=200 checkpoint.every=20
```
