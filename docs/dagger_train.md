### 数据清晰
*检测穿模*
```bash
python -u -m grail.datatool.batch_render_replay_clip \
  --data_dir data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded \
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
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint \
  headless=True \
  num_envs=8


python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs \
  headless=False \
  num_envs=8
  
```

## 服务器端训练
```bash
cd /home/GRAIL/imports/SONIC

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint \
  headless=True \
  num_envs=4096 \
  experiment_name=mlp_bc1_ppo1_studentonly_cleandata_joint_ema \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.num_learning_iterations=10000 \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt 

CUDA_VISIBLE_DEVICES=1 python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint \
  headless=True \
  num_envs=4096 \
  experiment_name=mlp_bc_ppo_studentonly_nonormalize \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.num_learning_iterations=10000 \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  algo.config.ppo_bc_loss_schedule.adaptive_after_iteration=1000


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
  ++manager_env.config.object_usd_path=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/
```
## 服务器端play
```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/mlp_bc1_ppo1_studentonly-20260902_174901/model_step_002000.pt \
  +headless=True \
  ++num_envs=32 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/mlp_bc1_ppo1_studentonly-20260902_174901 \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
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
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
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
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
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

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODIFIED STUDENT ACTOR                                          │
│                  Joint-Normalization Direct-Concatenation MLP Policy                         │
├──────────────────────────────────────────────┬───────────────────────────────────────────────┤
│ obs["proprio_obs"]                           │ obs["privileged_obs"]                         │
│ shape = [B,5,161]                            │ shape = [B,48]                                │
│                                              │                                               │
│ frame t-4 ─┐                                 │ object_bps──────────────────────10 ─┐         │
│ frame t-3 ─┤                                 │ table_corners_b─────────────────12 ─┤         │
│ frame t-2 ─┼─ joint_pos──────────────43      │ object_pos_b────────────────────3 ─┤         │
│ frame t-1 ─┤  joint_vel──────────────43      │ object_ori_b_6d─────────────────6 ─┼─concat─┐│
│ frame t   ─┘  projected_gravity───────3      │ hand_object_transform_6d────────9 ─┤        ││
│                base_ang_vel────────────3      │ contact_force_magnitude─────────8 ─┘        ││
│                base_lin_vel────────────3      │                                              ││
│                last_meta_action────────66     │                                              ││
│                                      ───      │                                              ││
│                         per-frame dim = 161   │                                              ││
│                                │             │                                              ││
│                 Flatten [B,5,161] → [B,805]  │                               [B,48] ◄───────┘│
│                                │             │                                  │            │
│                                └──────────────────────┬───────────────────────────┘            │
│                                                       │                                        │
│                                      DIRECT CONCATENATION                                      │
│                                                       │                                        │
│                 xraw = cat([flatten(proprio_obs), privileged_obs], dim=-1)                     │
│                                                       │                                        │
│                                             xraw: [B,853]                                      │
│                                                       │                                        │
│                         ┌─────────────────────────────▼─────────────────────────────┐          │
│                         │          SHARED / JOINT EMA STANDARDIZATION              │          │
│                         │                                                           │          │
│                         │ Running statistics over the complete 853-D input:         │          │
│                         │                                                           │          │
│                         │   μjoint ∈ R^853          varjoint ∈ R^853                │          │
│                         │   σjoint = sqrt(max(varjoint, ε²))                        │          │
│                         │   ε = 1.0e−4                                              │          │
│                         │                                                           │          │
│                         │   xnorm = (xraw − μjoint) / σjoint                        │          │
│                         │   xnorm = clamp(xnorm, −5, +5)                            │          │
│                         │                                                           │          │
│                         │ EMA update after PPO epochs:                              │          │
│                         │   μjoint ← (1−m)μjoint + m·μbatch                         │          │
│                         │   varjoint ← (1−m)varjoint + m·varbatch                   │          │
│                         │   momentum m = 0.05                                       │          │
│                         └─────────────────────────────┬─────────────────────────────┘          │
│                                                       │                                        │
│                                                xnorm: [B,853]                                  │
│                                                       │                                        │
│                                      ┌────────────────▼────────────────┐                       │
│                                      │       Linear(853 → 2048)        │                       │
│                                      │ params = 853×2048 + 2048        │                       │
│                                      │         = 1,748,992             │                       │
│                                      └────────────────┬────────────────┘                       │
│                                                       │                                        │
│                                                     SiLU                                       │
│                                                       │                                        │
│                                      ┌────────────────▼────────────────┐                       │
│                                      │      Linear(2048 → 1800)        │                       │
│                                      │ params = 2048×1800 + 1800       │                       │
│                                      │         = 3,688,200             │                       │
│                                      └────────────────┬────────────────┘                       │
│                                                       │                                        │
│                                                     SiLU                                       │
│                                                       │                                        │
│                                      ┌────────────────▼────────────────┐                       │
│                                      │       Linear(1800 → 512)        │                       │
│                                      │ params = 1800×512 + 512         │                       │
│                                      │         = 922,112               │                       │
│                                      └────────────────┬────────────────┘                       │
│                                                       │                                        │
│                                                     SiLU                                       │
│                                                       │                                        │
│                                      ┌────────────────▼────────────────┐                       │
│                                      │        Linear(512 → 66)         │                       │
│                                      │ params = 512×66 + 66            │                       │
│                                      │         = 33,858                │                       │
│                                      └────────────────┬────────────────┘                       │
│                                                       │                                        │
│                                  predicted normalized meta-action                              │
│                                              ânorm: [B,66]                                     │
│                                                       │                                        │
│                     â = clip(ânorm,−5,+5) ⊙ sqrt(target_var) + target_mean                     │
│                                                       │                                        │
│                                          action_mean μ: [B,66]                                │
│                                                       │                                        │
│                      ┌────────────────────────────────┴──────────────────────────────┐         │
│                      │                                                               │         │
│                      ▼                                                               ▼         │
│          decoder latent: [B,64]                                      hand primitives: [B,2]   │
│          ├─ latent[0:64]                                             ├─ left hand primitive   │
│          └─ direct pre-FSQ latent                                    └─ right hand primitive  │
│                      │                                                               │         │
│                      └────────────────────────────────┬───────────────┘                         │
│                                                       │                                        │
│                                      trainable log_std parameter [66]                          │
│                                                       │                                        │
│                                std = exp(clamp(log_std, −20, 2))                               │
│                                                       │                                        │
│                              πstudent(a|o) = Normal(μ, diag(std²))                             │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Joint-normalizer buffers : mean[853] + variance[853] + update counter                         │
│ MLP parameters           : 6,393,162                                                           │
│ Gaussian log_std         :        66                                                           │
│ Student trainable total  : 6,393,228                                                           │
│ Original actor total     : 6,393,988                                                           │
│ Parameter difference     :      −760  (−0.012%)                                                │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```
