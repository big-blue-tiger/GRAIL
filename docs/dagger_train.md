### 数据清理
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

训练会自动读取 `motion_file` 所在数据集的 `meta/<motion_key>.pkl`。
若包含 `transition_start_frame` 和 `transition_end_frame`，则与 play 使用同一套残差过渡：
起始帧之前为纯 SONIC，过渡区间内 GRAIL 残差系数从 0 线性递增至 1，结束帧之后为 SONIC + 完整 GRAIL 残差。
系数按各环境当前 motion 的实际帧计算，包含随机起始偏移与源数据帧率换算；没有过渡元数据的 motion 保持原行为。
DAgger 的教师执行动作和蒸馏目标均应用此系数（包括 action-chunk）；学生的完整 latent 输出通过该目标学习，不直接乘残差系数。
使用拼接数据时，将训练命令的 `motion_file`、`object_motion_file`、`bps_dir` 和 `object_usd_path` 指向拼接数据集，并保留同级 `meta` 目录即可，无需额外开关。
若希望每次都从行走第一帧开始，再加 `++manager_env.commands.motion.start_from_first_frame=True` 和 `++manager_env.commands.motion.sample_before_contact=False`。

`joint_micro_step` 启用 `algo.config.teacher_termination_loss_override`：超过教师终止阈值、但尚未超过学生终止阈值的状态不参与 BC loss。BC loss 按剩余有效样本数归一化，再乘基础调度的 BC 系数；全部被屏蔽时 loss 为零。PPO/BC 调度系数不随接触标签改变。

`joint_micro_step` 继承的 `action_acc_l2` 对 `joint_pos.processed_actions`（已包含逐关节缩放、偏置和裁剪的目标角度）计算二阶差分，再除以控制周期 `step_dt` 的平方，最后平方求和。它与实际关节加速度惩罚 `joint_acc_l2` 具有相同单位，两项初始权重均为 `-2.5e-7`；这不保证实际奖励贡献相等。
若需平均贡献接近 1:1，在同一训练窗口读取 `Episode_Reward/action_acc_l2` 和 `Episode_Reward/joint_acc_l2`，分别取平均值的绝对值 `P_action`、`P_joint`，使用 `w_action_new = w_action_current * P_joint / P_action` 校准，并在后续 rollout 复核。两项日志已包含权重和相同的时间归一化，无需再次乘权重；`P_action` 接近零时不能使用此比例。目标角度跳变、接触冲击和 PD 跟踪误差会使比例随策略变化。

```bash
cd /home/tide/robot/GRAIL/imports/SONIC

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint_micro_step \
  headless=True \
  num_envs=16 \
  ++manager_env.config.gpu_collision_stack_size_exp=28 \
++algo.config.num_mini_batches=1


python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/imports/SONIC/gear_sonic/config/exp/manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint_final_goal_hand_near.yaml \
  headless=False \
  num_envs=8
  
```

### 本地直接replay

修改单条 Kimodo 约束后，可一键上传、远程生成、下载覆盖、拼接并重新渲染：

```bash
bash grail/walk_data_tool/regenerate_kimodo_walk.sh \
    data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/kimodo_end_frame/pickup_table__alcohol_1__000.json
```

追加 `--dry-run` 可先检查匹配的生成命令和输出路径，不连接服务器、不修改数据。
不传 JSON 时默认处理上述动作。脚本从 JSON 同目录的 `generate_commands.sh`
精确匹配 `--constraints`，保留该动作的文本、时长和种子，仅改写远程输入/输出路径。
服务器使用实际容器 `yuguanchao_kimodo`，每次工作目录为
`/home/cuixinru/data0/kimodo/data/grail_single/<动作名>/<时间戳_PID>/`。
回传的同名 CSV/NPZ 覆盖 `kimodo_walk/`，随后仅拼接该动作（过渡 10 帧），
视频覆盖 `pickup_table_walk_concat/vis/<动作名>.mp4`。任一步失败即停止；
渲染成功后才替换旧视频，其他动作的视频和已有合集不会更新。

批量 replay：

```bash
CUDA_VISIBLE_DEVICES=0 bash grail/visualization/scripts/visualize.sh \
    data/hf_dataset/data_update/data/pickup_table_walk_concat \
    16 1.5,-1.5,1.0 xyzw local 1
```

## 服务器端训练
```bash
cd /home/GRAIL/imports/SONIC

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint_micro_step \
  headless=True \
  num_envs=1024 \
  experiment_name=mlp_normalize_low_lr \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  algo.config.num_learning_iterations=10000 

CUDA_VISIBLE_DEVICES=2 python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint_micro_step \
  headless=True \
  num_envs=1024 \
  experiment_name=joint_micro_step_high_regularization \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  algo.config.num_learning_iterations=10000 \
  algo.config.num_mini_batches=4 \
  algo.config.ppo_bc_loss_schedule.adaptive_after_iteration=1000

CUDA_VISIBLE_DEVICES=1 python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint_micro_step \
  headless=True \
  num_envs=4096 \
  experiment_name=0308Termination_walk \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  algo.config.num_learning_iterations=15000 \
  algo.config.ppo_bc_loss_schedule.adaptive_after_iteration=1000 \
  algo.config.ppo_bc_loss_schedule.bc_min_coef=0.05

CUDA_VISIBLE_DEVICES=1 python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint_micro_step \
  headless=True \
  num_envs=1024 \
  experiment_name=walk01bccoef00shiftbc \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  algo.config.num_learning_iterations=10000 \
  algo.config.num_mini_batches=4 \
  algo.config.ppo_bc_loss_schedule.adaptive_after_iteration=1000 \
  algo.config.ppo_bc_loss_schedule.bc_min_coef=0.1
```

## 本机play
```bash
cd /home/tide/robot/GRAIL/imports/SONIC

python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/tide/robot/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint_micro_step-20260915_160202/last.pt \
  +headless=True \
  ++num_envs=8 \
  ++manager_env.config.gpu_collision_stack_size_exp=28 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/tide/robot/GRAIL/outputs/robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint_micro_step \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/
```
## 纯PNPTABLE PLAY

Play 自动读取动作对应的 `meta/<motion_key>.pkl`。若同时包含
`transition_start_frame` 和 `transition_end_frame`，latent 残差会额外乘以
`clamp((f - start) / (end - start), 0, 1)`，其中 `f` 按原始动作 FPS
从当前仿真步数换算（包含 reset 的起始偏移）。起始帧系数为 0，结束帧为 1，
结束后恢复原有 `latent_residual_scale`；缺少任一字段则保持原行为。
两个字段须为从 0 开始的整数，且 `0 <= start < end`。
仅缩放 latent 残差，不改变手指 primitive 输出；训练和 direct-latent 学生不受影响。

```bash
python gear_sonic/eval_agent_trl.py \
  --config-name=base \
  +exp=manager/universal_token/hoi/pnp_table \
  +callbacks=im_eval \
  checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  headless=True \
  num_envs=15 \
  ++run_once=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
  ++manager_env.commands.motion.start_from_first_frame=True \
  ++manager_env.commands.motion.sample_before_contact=False \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  ++manager_env.config.gpu_collision_stack_size_exp=29 \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/pnp_table_walk_concat_video \
  ++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg \
  ++manager_env.recorders.render_envs.video_save_path=/home/GRAIL/outputs/pnp_table_walk_concat_video \
  ++manager_env.recorders.render_envs.video_quality=5
```
### Teacher policy 检测整套数据成功率（不保存视频）

每条动作从第一帧直接运行 teacher policy，只统计首个 episode：出现 timeout 即成功，
其他提前终止计为失败；同一步同时出现 timeout 和其他终止条件也计为 timeout 成功。
结束时打印 `Timeout Success Rate: 成功数/总数 = 百分比`，JSON 的
`summary.timeout_success_rate` 保存 0–1 成功率（未全部完成时为 `null`）。
原有 `accepted` 是数据清理的更严格判定，不用于此成功率。

`num_envs` 自动取 `robot/` 下的动作数（当前为 69），保证每条数据检测一次。
不能直接改成较小的环境数，否则 `run_once` 只检测前 `num_envs` 条；
数据量较大时应使用独立进程分片，避免同一场景切换动作后物体碰撞模型不匹配。

```bash
cd /home/GRAIL/imports/SONIC

TEACHER_EVAL_DATA=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat
TEACHER_EVAL_ENVS=$(find "$TEACHER_EVAL_DATA/robot" -type f -name '*.pkl' | wc -l)

python gear_sonic/eval_agent_trl.py \
  --config-name=base \
  +exp=manager/universal_token/hoi/pnp_table \
  +callbacks=im_eval \
  checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  headless=True \
  num_envs="$TEACHER_EVAL_ENVS" \
  ++run_once=True \
  '++eval_callbacks=[]' \
  ++run_once_report_path=/home/GRAIL/outputs/teacher_walk_concat_success.json \
  ++manager_env.commands.motion.init_z_offset=0.05 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
  ++manager_env.commands.motion.start_from_first_frame=True \
  ++manager_env.commands.motion.sample_before_contact=False \
  ++manager_env.config.object_usd_path="$TEACHER_EVAL_DATA/object_usd" \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="$TEACHER_EVAL_DATA/robot" \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file="$TEACHER_EVAL_DATA/objects" \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir="$TEACHER_EVAL_DATA/bps" \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  ++manager_env.config.gpu_collision_stack_size_exp=30 \
  ++manager_env.config.render_results=False \
  '++manager_env.recorders={}'
```

使用已配置 Isaac Lab 的 Python 环境运行。若要打开窗口观察，将 `headless=True` 改为
`headless=False`，仍不保存视频。终止阈值沿用原 play 命令的默认设置。

## 服务器端play
```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/0308Termination_walk-20260920_115339/last.pt \
  +headless=True \
  ++num_envs=16 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/0308Termination_walk-3200 \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  ++object_pos_deviation_threshold=25 \
&& \
python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/0308Termination_walk-20260920_115339/model_step_002000.pt \
  +headless=True \
  ++num_envs=16 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/0308Termination_walk-model_step_002000 \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  ++object_pos_deviation_threshold=25 



python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/joint_micro_step_high_regularization-20260915_162103/last.pt \
  +headless=True \
  ++num_envs=16 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/joint_micro_step_high_regularization-20260915_162103 \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  ++object_pos_deviation_threshold=25
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
