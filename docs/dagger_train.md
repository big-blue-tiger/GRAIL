
### 在本地训练

```bash
cd /home/tide/robot/GRAIL/imports/SONIC

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_diffusion_decoder_latent_vector_obs \
  headless=False \
  num_envs=8


python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_transformer_flow_chunk40_decoder_latent_vector_obs_linear_dagger \
  headless=False \
  num_envs=8
  
```

## 服务器端训练
```bash
cd /home/GRAIL/imports/SONIC

CUDA_VISIBLE_DEVICES=1 python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_diffusion_decoder_latent_vector_obs \
  headless=True \
  num_envs=1024 \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt 

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_transformer_flow_chunk40_decoder_latent_vector_obs \
  headless=True \
  num_envs=1024 \
  experiment_name=chunk40_baseline \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  algo.config.action_chunk.train_micro_batch_size=64

CUDA_VISIBLE_DEVICES=1 python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_transformer_flow_chunk40_decoder_latent_vector_obs_linear_dagger \
  headless=True \
  num_envs=1024 \
  experiment_name=chunk40_linear_dagger \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  algo.config.action_chunk.train_micro_batch_size=64

CUDA_VISIBLE_DEVICES=2 python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_transformer_flow_chunk1_decoder_latent_vector_obs \
  headless=True \
  num_envs=1024 \
  experiment_name=chunk1 \
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
python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/manager/universal_token/distill/0821_teacher_student_rollout/last.pt \
  +headless=True \
  ++num_envs=4 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/0821_teacher_student_rollout \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/

python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/chunk40_baseline-20260823_182155/last.pt \
  +headless=True \
  ++num_envs=8 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/0821_with_only_student \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/



  python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/chunk40_linear_dagger-20260823_183856/last.pt \
  +headless=True \
  ++num_envs=16 \
  +run_once=True \
  ++object_pos_deviation_threshold=25.0 \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/chunk40_linear_dagger-20260823_183856 \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/

    python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/chunk1-20260823_173553/last.pt \
  +headless=True \
  ++num_envs=8 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/chunk1-20260823_173553 \
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

 proprio_obs [B,5,161]
          │
        Flatten
          │
       [B,805]
          │
   Linear 805→1024
          │
         SiLU
          │
   Linear 1024→512
          │
          └──────────────────────────────► ROBOT TOKEN ───────┐
                                                              │
 privileged_obs [B,48]                                        │
          │                                                   │
    Linear 48→512                                             │
          │                                                   │
         SiLU                                                 │
          │                                                   │
    Linear 512→512                                            │
          │                                                   │
          └─────────────────────────────► OBJECT TOKEN ────────┤
                                                              │
 flow timestep t                                              │
          │                                                   │
  Sin/Cos Embedding 256D                                      │
          │                                                   │
    Linear 256→512                                            │
          │                                                   │
         SiLU                                                 │
          │                                                   │
    Linear 512→512                                            │
          │                                                   │
          └──────────────────────────────► TIME TOKEN ─────────┤
                                                              │
                                                              ▼
                                                    [OBJECT, ROBOT, TIME]
                                                         [B,3,512]
                                                              │
                                                              │
 noisy action chunk [B,40,66]                                 │
          │                                                   │
    Linear 66→512                                             │
          │                                                   │
    + Action Position                                         │
          │                                                   │
          ▼                                                   │
      [B,40,512]                                              │
          │                                                   │
          ├───────────────────────────────────────────────────┘
          │
          ▼
 ╔══════════════════════════════════════════════════════════════╗
 ║                     RDT BLOCK × 14                         ║
 ║                                                              ║
 ║  RMSNorm                                                     ║
 ║     ↓                                                        ║
 ║  8-head Self-Attention          40 action tokens ↔ 40        ║
 ║     ↓ + residual                                             ║
 ║  RMSNorm                                                     ║
 ║     ↓                                                        ║
 ║  8-head Cross-Attention         40 action tokens → 3 cond    ║
 ║     ↓ + residual                                             ║
 ║  RMSNorm                                                     ║
 ║     ↓                                                        ║
 ║  FFN 512 → 512 → 512                                        ║
 ║     ↓ + residual                                             ║
 ╚══════════════════════════════╤═══════════════════════════════╝
                                │
                                ▼
                          [B,40,512]
                                │
                             RMSNorm
                                │
                         Linear 512→66
                                │
                                ▼
                    predicted velocity [B,40,66]

