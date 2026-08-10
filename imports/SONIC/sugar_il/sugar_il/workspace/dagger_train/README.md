# Online DAgger

```bash
conda activate grail
python -m sugar_il.workspace.dagger_train.train \
  paths.input=../../../data/hf_dataset/data_update/data/pickup_table/robot \
  paths.teacher_checkpoint=../models/pnp_table/last.pt \
  paths.generator_checkpoint=data/outputs/2026.08.02/17.35_train_generator_ObjectAwareSONIC/checkpoints/epoch-0002-val_loss-0.054.ckpt \
  paths.output_dir=data/outputs/dagger_online_envs1024 \
  training.teacher_ratio=0.5 \
  training.learning_rate=2e-4 \
  --headless \
  environment.max_parallel_envs=1024 \
  training.batch_size=128 \
 
```

