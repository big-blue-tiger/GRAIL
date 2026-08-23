
### 在本地训练

```bash
cd /home/tide/robot/GRAIL/imports/SONIC

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_diffusion_decoder_latent_vector_obs \
  headless=False \
  num_envs=8


python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_transformer_flow_chunk40_decoder_latent_vector_obs \
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
  ++num_envs=16 \
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
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/manager/universal_token/distill/0821_with_only_student/last.pt \
  +headless=True \
  ++num_envs=16 \
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


========================================================================================================
                    Dual-Encoder Vector Diffusion Policy — Modified Architecture
========================================================================================================

                                             student_obs
                                                  │
                     ┌────────────────────────────┴────────────────────────────┐
                     │                                                         │
                     ▼                                                         ▼
       ┌────────────────────────────────────┐                    ┌─────────────────────────────────────┐
       │     PROPRIOCEPTIVE FEATURES        │                    │       PRIVILEGED FEATURES           │
       │         本体状态 / 8-frame         │                    │          当前帧                      │
       │                                    │                    │                                     │
       │ per frame:                         │                    │ object_bps                 10D       │
       │ joint_pos                  43D      │                    │ table_corners_b            12D       │
       │ projected_gravity           3D      │                    │   4 corners × xyz                     │
       │ base_ang_vel                3D      │                    │   relative to pelvis frame            │
       │ base_lin_vel                3D      │                    │ object_pos_b                3D       │
       │ ───────────────────────────────    │                    │ object_ori_b_6d             6D       │
       │ single frame               52D      │                    │ hand_object_transform_6d     9D       │
       │                                    │                    │ contact_force_magnitude      8D       │
       │ history = 8 frames                  │                    │ ───────────────────────────────     │
       │                                    │                    │ total                      48D       │
       │ 8 × 52 = 416D                       │                    │                                     │
       │                                    │                    │ shape: [B,48]                       │
       │ shape: [B,8,52] → [B,416]          │                    └─────────────────┬───────────────────┘
       └─────────────────┬──────────────────┘                                      │
                         │                                                         │
                         ▼                                                         ▼
          ╔════════════════════════════╗                             ╔════════════════════════════╗
          ║   PROPRIO ENCODER MLP      ║                             ║  PRIVILEGED ENCODER MLP    ║
          ║                            ║                             ║                            ║
          ║ Linear 416 → 512           ║                             ║ Linear 48 → 256            ║
          ║ SiLU                       ║                             ║ SiLU                       ║
          ║ Linear 512 → 256           ║                             ║ Linear 256 → 256           ║
          ║                            ║                             ║                            ║
          ║ output: [B,256]            ║                             ║ output: [B,256]            ║
          ╚═════════════╤══════════════╝                             ╚═════════════╤══════════════╝
                        │                                                          │
                        │ proprio_feat                                             │ privileged_feat
                        │ [B,256]                                                  │ [B,256]
                        │                                                          │
                        └─────────────────────────┬────────────────────────────────┘
                                                  │
                                                  │
                         noisy latent x_t         │            timestep t
                              [B,66]              │               [B]
                                 │                │                │
                                 │                │                ▼
                                 │                │      ┌──────────────────────┐
                                 │                │      │ SinusoidalPosEmb     │
                                 │                │      │      128D            │
                                 │                │      │        ↓             │
                                 │                │      │ Linear 128 → 512     │
                                 │                │      │ SiLU                 │
                                 │                │      │ Linear 512 → 128     │
                                 │                │      └──────────┬───────────┘
                                 │                │                 │
                                 │                │              [B,128]
                                 │                │                 │
                                 └────────────────┼─────────────────┘
                                                  │
                                                  ▼
                ┌───────────────────────────────────────────────────────────────────┐
                │                         CONCATENATE                               │
                │                                                                   │
                │ noisy latent x_t             66D   当前 flow latent              │
                │ proprio feature             256D   8帧机器人本体历史             │
                │ privileged feature          256D   当前物体/桌面/接触信息         │
                │ timestep embedding          128D   flow time                     │
                │ ───────────────────────────────────────────────────────────────   │
                │ TOTAL                        706D                                  │
                │                                                                   │
                │                         shape [B,706]                              │
                └────────────────────────────────┬──────────────────────────────────┘
                                                 │
                                                 ▼
                            ╔══════════════════════════════════════╗
                            ║          DENOISER MLP               ║
                            ╠══════════════════════════════════════╣
                            ║                                      ║
                            ║ Linear 706  → 1024 + SiLU            ║
                            ║ Linear 1024 → 1024 + SiLU            ║
                            ║ Linear 1024 →  512 + SiLU            ║
                            ║ Linear 512  →   66                   ║
                            ║                                      ║
                            ║ output: predicted velocity [B,66]    ║
                            ╚═══════════════════╤══════════════════╝
                                                │
                                                │ 4-step Flow Matching integration
                                                ▼
                                         66D meta action
                                  ┌─────────────┴─────────────┐
                                  │                           │
                              64D latent                  2D hand primitive
                                  │                           │
                                  ▼                           ▼
                         Frozen SONIC Decoder           Finger Primitive Map
                                  │                           │
                             29D body action              14D hand action
                                  └─────────────┬─────────────┘
                                                ▼
                                      G1 final action [B,43]