
### 在本地训练

```bash
cd /home/tide/robot/GRAIL/imports/SONIC

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_diffusion_decoder_latent_vector_obs \
  headless=False \
  num_envs=8
```

## 服务器端训练
```bash
cd /home/GRAIL/imports/SONIC

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_diffusion_decoder_latent_vector_obs \
  headless=True \
  num_envs=1024 \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt

```

## 本机play
```bash
cd /home/tide/robot/GRAIL/imports/SONIC


python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/tide/robot/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/manager/universal_token/distill/robocasa_pickup_table_diffusion_decoder_latent_vector_obs_robocasa_pickup_table_diffusion_decoder_latent_vector_obs-20260820_175517/last.pt \
  +headless=True \
  ++num_envs=8 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/tide/robot/GRAIL/outputs/student_play_16env \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/
```
## 服务器端play
```bash
CUDA_VISIBLE_DEVICES=1 python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/manager/universal_token/distill/robocasa_pickup_table_diffusion_decoder_latent_vector_obs_robocasa_pickup_table_diffusion_decoder_latent_vector_obs-20260820_175517/last.pt \
  +headless=True \
  ++num_envs=16 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/student_play_8env \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/
```
```bash
rsync -aP \
  --no-owner --no-group \
  --exclude='data/' \
  -e "ssh -p 9991 -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
  /home/tide/robot/GRAIL/ \
  ygc@202.120.37.249:/home/ygc/data0/GRAIL/


  rsync -aP \
  --no-owner --no-group \
  -e "ssh -p 9991 -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
  /home/tide/robot/GRAIL/imports/ \
  ygc@202.120.37.249:/home/ygc/data0/GRAIL/imports/
```