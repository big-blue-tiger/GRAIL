### 数据清晰
*检测穿模*
```bash
python -u -m grail.datatool.batch_render_replay_clip \
  --data_dir data/hf_dataset/data_update/data/pickup_table \
  --output_dir data/hf_dataset/data_update/data/pickup_table_cleaned \
  --quat_convention xyzw \
  --no_record_video
```

*检测是否成功*
```bash
python -u -m grail.datatool.batch_filter_teacher_policy \
  --data_dir data/hf_dataset/data_update/data/pickup_table_cleaned \
  --output_dir data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded \
  --teacher_checkpoint imports/SONIC/models/pnp_table/last.pt \
  --num_envs 512 \
  --no_record_video
```

### 在本地训练

```bash
cd /home/tide/robot/GRAIL/imports/SONIC

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_privileged_history \
  headless=True \
  num_envs=8


python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_transformer_flow_chunk40_decoder_latent_vector_obs_linear_dagger \
  headless=False \
  num_envs=8
  
```

## 服务器端训练
```bash
cd /home/GRAIL/imports/SONIC

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_privileged_history \
  headless=True \
  num_envs=4096 \
  experiment_name=mlp_bc1_ppo1_studentonly_cleandata_prihistory \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.num_learning_iterations=10000 \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt 

CUDA_VISIBLE_DEVICES=1 python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs \
  headless=True \
  num_envs=4096 \
  experiment_name=mlp_bc1_ppo1_studentonly \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.num_learning_iterations=10000 \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  algo.config.dagger_student_ratio=1.0

cd /home/GRAIL/imports/SONIC

CUDA_VISIBLE_DEVICES=0,1,2 \
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=3 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_privileged_history \
  headless=True \
  num_envs=2400 \
  experiment_name=mlp_bc1_ppo1_studentonly_cleandata_prihistory \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.num_learning_iterations=10000 \
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
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/mlp_bc1_ppo1_studentonly-20260828_161529/last.pt \
  +headless=True \
  ++num_envs=24 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/mlp_bc1_ppo1_studentonly-20260828_161529 \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  ++object_pos_deviation_threshold=25.0

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
  --exclude='data/' \
  --exclude='imports/blender/' \
  --exclude='outputs/' \
  -e "ssh -p 9989 -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
  /home/tide/robot/GRAIL/ \
  cuixinru@202.120.37.249:/home/cuixinru/data0/GRAIL/
```

B = rollout 时 4096；训练时 B 为 minibatch 大小
┌────────────────────────────────── STUDENT ACTOR ──────────────────────────────────┐
│                                                                                    │
│  proprio_obs[t-4:t] : [B,5,161]                                                    │
│  每帧 161D                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │ joint_pos 43 │ joint_vel 43 │ gravity 3 │ ωbase 3 │ vbase 3 │ prev_action 66 │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                 │ Flatten history                                                   │
│                 ▼                                                                   │
│              [B,805]                                                                │
│                 │ EMA Standardize: (x-μp)/√(σp²+1e-4²), clip[-5,5], momentum=.05    │
│                 ▼                                                                   │
│      Linear(805→1024) + SiLU                  参数: 825,344                          │
│                 │                                                                   │
│      Linear(1024→512), no activation          参数: 524,800                          │
│                 │                                                                   │
│                 └──────────────────────► proprio_feature [B,512] ───────────┐        │
│                                                                            │        │
│  privileged_obs[t] : [B,48]                                                │        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │        │
│  │ object BPS 10 │ table corners 12 │ object pose 3+6 │ hand↔object 9   │ │        │
│  │ contact-force magnitudes 8                                           │ │        │
│  └───────────────────────────────────────────────────────────────────────┘ │        │
│                 │ EMA Standardize, clip[-5,5], momentum=.05                │        │
│                 ▼                                                          │        │
│       Linear(48→512) + SiLU                   参数: 25,088                  │        │
│                 │                                                          │        │
│       Linear(512→512), no activation          参数: 262,656                 │        │
│                 │                                                          │        │
│                 └────────────────────► privileged_feature [B,512] ─────────┤        │
│                                                                            ▼        │
│                                                        Concat [512‖512] = [B,1024]   │
│                                                                            │        │
│                                      ┌─────────────────────────────────────┘        │
│                                      ▼                                              │
│                            Linear(1024→2048) + SiLU       参数: 2,099,200            │
│                                      │                                              │
│                            Linear(2048→1024) + SiLU       参数: 2,098,176            │
│                                      │                                              │
│                            Linear(1024→512)  + SiLU       参数:   524,800            │
│                                      │                                              │
│                            Linear(512→66), no activation  参数:    33,858            │
│                                      │                                              │
│                                      ▼                                              │
│                         normalized prediction â_norm [B,66]                          │
│                                      │                                              │
│             target EMA de-normalization: μ = â_norm·√var_action + mean_action       │
│                                      │                                              │
│                    ┌─────────────────┴──────────────────┐                           │
│                    │                                    │                           │
│                    ▼                                    ▼                           │
│          μlatent [B,64]                         μhand [B,2]                          │
│      full pre-quantization latent          left/right primitive mean                │
│                    └─────────────────┬──────────────────┘                           │
│                                      │                                              │
│           trainable state-independent log_std[66], init log(0.01)                   │
│                                      │ exp                                           │
│                                      ▼                                              │
│                  πstudent(a|s)=Normal(μ(s), diag(σ²)),  σ∈R66                       │
│                                      │                                              │
│                  rollout: sample a [B,66]   evaluation: usually μ                   │
│                                                                                    │
│  ───────────────────────────── TRAINING-ONLY PATH ───────────────────────────────  │
│                                                                                    │
│  external label a* [B,66] ──► action EMA standardize ──► a*norm                    │
│                                                          │                         │
│  â_norm ─────────────────────────────────────────────────┼─► MSE(â_norm,a*norm)     │
│                                                          │       × bc_loss_coef=10  │
│                                                          └───────× λBC(k)           │
│                                                                                    │
│  sampled action/logπ/advantage ─────────────────────────────────► λPPO(k)·L_PPO     │
└────────────────────────────────────────────────────────────────────────────────────┘

策略边界之外的执行语义：
  action[:64] ─► ATM/VQ quantize ─► frozen decoder ─► 29-DOF body action sequence
  action[64:] ─► threshold/primitive map           ─► left/right finger targets