## 推理
```bash
python -u gear_sonic/eval_agent_trl.py \
  ++checkpoint=/home/tide/robot/GRAIL/imports/SONIC/models/terrain_release/last.pt \
  ++headless=False \
  ++num_envs=4 \
  ++use_wandb=False \
  '++eval_callbacks=[]' \
  ++run_eval_loop=True \
  ++manager_env.config.render_results=False \
  ++manager_env.recorders.render_envs=null \
  ++algo.config.eval.save_videos=False \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/tide/robot/GRAIL/data/hf_dataset/sitting/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/tide/robot/GRAIL/data/hf_dataset/sitting/objects \
  ++manager_env.config.terrain_motion_dir=/home/tide/robot/GRAIL/data/hf_dataset/sitting \
  ++manager_env.config.flat_usd_path=/home/tide/robot/GRAIL/data/hf_dataset/sitting/object_usd/sitting__acadian_rustic_wooden_dining_chair_distressed_white__000.usd \
  ++manager_env.config.flat_to_terrain_ratio=0 \
  ++manager_env.commands.motion.flat_to_terrain_ratio=0 \
  '++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[sitting__acadian_rustic_wooden_dining_chair_distressed_white__000]' \
  ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
  ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=1
```