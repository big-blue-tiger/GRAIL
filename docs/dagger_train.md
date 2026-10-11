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

```bash
cd /home/tide/robot/GRAIL/imports/SONIC
WANDB_MODE=disabled python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_place_mlp_decoder_latent_vector_obs_joint_micro_step \
  headless=True num_envs=4 experiment_name=place_migration_smoke \
  ++manager_env.config.gpu_collision_stack_size_exp=26 \
  algo.config.num_mini_batches=1 algo.config.num_learning_epochs=1 \
  algo.config.num_learning_iterations=50 \
  manager_env.commands.motion.motion_lib_cfg.max_unique_motions=2 \
  ++manager_env.commands.motion.start_from_first_frame=True \
  ++manager_env.commands.motion.sample_before_contact=False
```

这只检查训练链路，随机初始化学生不保证完整走到释放段。新 place 实验从头训练，
保留继承配置中的正常 latent moments 统计和冻结调度。普通续训本 place 实验使用 `resume=True`。

可选 `manager_env.recorders.object_position_success.diagnostic_path=/absolute/path/trace.jsonl`
只用于小规模完整 rollout，逐步记录源阶段、参考 contact、实际接触 link 数、目标距离及所有奖励的
实际加权贡献。`weighted_reward_rates` 尚未乘 `step_dt`；累计控制步贡献需乘 trace 的 `step_dt`。
阶段编码 0/1/2/3 对应首次参考接触前、原 GRAIL 抓握、新增搬运、最终释放；该 trace 不改变采样或奖励。

### 本地直接replay

修改单条 Kimodo 约束后，可一键上传、远程生成、下载覆盖、拼接并重新渲染：

```bash
bash grail/walk_data_tool/regenerate_kimodo_walk.sh \
    data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/kimodo_end_frame/pickup_table__alcohol_1__000.json
```

批量 replay：

```bash
CUDA_VISIBLE_DEVICES=0 bash grail/visualization/scripts/visualize.sh \
    data/hf_dataset/data_update/data/pickup_table_place_concat \
    32 1.5,-1.5,1.0 xyzw local 1
```

## 服务器端训练

训练和 play 自动循环（先激活 Isaac Lab Python 环境，并登录 W&B）：

```bash
cd /home/GRAIL/imports/SONIC
bash gear_sonic/scripts/train_play_grasp_walk_1008.sh --delay-hours 8 --delay-minutes 5
# 只查看实际命令，不等待、不训练：
bash gear_sonic/scripts/train_play_grasp_walk_1008.sh --dry-run
```

不传倒计时参数则立即开始。总训练长度保持 15000，分别在 3000、6000、9000、12000、15000
暂停训练并生成视频；前四次 play 完成后，从该阶段最新的编号检查点 resume。
实验目录只创建一次，W&B 使用该目录 `meta.yaml` 中的同一个 run ID。
视频保存在 `/home/GRAIL/outputs/<本次实验目录名>/iteration_<六位iteration>/`，
训练和 play 日志保存在实验目录的 `train_play_logs/`。任一阶段失败时脚本停止。
服务器需同步脚本、`segment_stop_callback.py` 和 `train_agent_trl.py` 的日志收尾修改。

```bash
cd /home/GRAIL/imports/SONIC

CUDA_VISIBLE_DEVICES=1  python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_place_mlp_decoder_latent_vector_obs_joint_micro_step \
  headless=True \
  num_envs=5 \
  experiment_name=mlp_normalize_low_lr \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  algo.config.num_learning_iterations=10000 

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_place_mlp_decoder_latent_vector_obs_joint_micro_step \
  headless=True \
  num_envs=4096 \
  experiment_name=GRASP_WALK_1009 \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  algo.config.num_learning_iterations=15000 \
  algo.config.ppo_bc_loss_schedule.adaptive_after_iteration=1000 \
  algo.config.ppo_bc_loss_schedule.bc_min_coef=0.04

```

## 服务器resume训练
```bash
WANDB_MODE=online python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_pickup_table_place_mlp_decoder_latent_vector_obs_joint_micro_step \
  headless=True \
  num_envs=4096 \
  experiment_name=GRASP_WALK_1009-resume4000 \
  project_name=GRAB_Tracking \
  resume=True \
  fork_resume=False \
  experiment_dir=logs_rl/GRAB_Tracking/GRASP_WALK_1009-resume4000 \
  checkpoint=logs_rl/GRAB_Tracking/GRASP_WALK_1009-20261009_070508/model_step_004000.pt \
  use_wandb=True \
  manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/object_usd \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/robot \
  manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/objects \
  manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/bps \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  algo.config.teacher_checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  algo.config.num_learning_iterations=15000 \
  algo.config.ppo_bc_loss_schedule.adaptive_after_iteration=1000 \
  algo.config.ppo_bc_loss_schedule.bc_min_coef=0.04

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
若包含 `residual_fade_out_start_frame` 和 `residual_fade_out_end_frame`，上述系数再乘以
`1 - clamp((f - start) / (end - start), 0, 1)`，在抓握后的拼接过渡内关闭残差，
第一帧 Kimodo 生成数据起保持为 0。渐出字段须成对提供且为整数，满足
`0 <= start < end < total_frames`；两个字段均缺失时渐出系数为 1，兼容旧数据。
play 和训练的教师执行及蒸馏目标共用这套系数，无需额外开关。
仅缩放 latent 残差，不改变手指 primitive 输出，也不直接缩放 direct-latent 学生动作。

```bash
#本地play
python gear_sonic/eval_agent_trl.py \
  --config-name=base \
  +exp=manager/universal_token/hoi/pnp_table \
  +callbacks=im_eval \
  checkpoint=/home/tide/robot/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  headless=True \
  num_envs=4 \
  ++run_once=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
  ++manager_env.commands.motion.start_from_first_frame=True \
  ++manager_env.commands.motion.sample_before_contact=False \
  ++manager_env.config.object_usd_path=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  ++manager_env.config.gpu_collision_stack_size_exp=27 \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/tide/robot/GRAIL/outputs/pnp_table_place_concat_video \
  ++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg \
  ++manager_env.recorders.render_envs.video_save_path=/home/tide/robot/GRAIL/outputs/pnp_table_place_concat_video \
  ++manager_env.recorders.render_envs.video_quality=5 \
  ++manager_env.terminations.ee_body_pos.params.threshold=25 \
  ++manager_env.terminations.object_pos_deviation.params.threshold=25

#服务器play
python gear_sonic/eval_agent_trl.py \
  --config-name=base \
  +exp=manager/universal_token/hoi/pnp_table \
  +callbacks=im_eval \
  checkpoint=/home/GRAIL/imports/SONIC/models/pnp_table/last.pt \
  headless=True \
  num_envs=16 \
  ++run_once=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1 \
  ++manager_env.commands.motion.start_from_first_frame=True \
  ++manager_env.commands.motion.sample_before_contact=False \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  ++manager_env.config.gpu_collision_stack_size_exp=29 \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/pnp_table_place_concat_video \
  ++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg \
  ++manager_env.recorders.render_envs.video_save_path=/home/GRAIL/outputs/pnp_table_place_concat_video \
  ++manager_env.recorders.render_envs.video_quality=5 \
  ++manager_env.terminations.ee_body_pos.params.threshold=25 \
  ++manager_env.terminations.object_pos_deviation.params.threshold=25 \
  ++manager_env.terminations.object_pos_deviation.params.threshold=25

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
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/GRASP_WALK_1009-20261009_070508/last.pt \
  +headless=True \
  ++num_envs=16 \
  +run_once=True \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/GRASP_WALK_1009-20261009_070508 \
  "~manager_env/recorders=empty" \
  "+manager_env/recorders=render" \
  ++manager_env.config.object_usd_path=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/objects \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat/bps \
  ++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/ \
  ++manager_env.terminations.ee_body_pos.params.threshold=25 \
  ++manager_env.terminations.object_pos_deviation.params.threshold=25 \
  ++manager_env.terminations.anchor_ori_full.params.threshold=2500

```

```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/03055Termination_walk-20260923_172120/last.pt \
  +headless=True ++num_envs=8 +run_once=True \
  ++manager_env.config.episode_length_s=13 \
  reference_free.enabled=True \
  reference_free.terminate_on_success=False \
  reference_free.dataset_dir=/home/GRAIL/data/SBTO/pickup_book/pickup_table \
  reference_free.initial_pose_file=/home/GRAIL/imports/SONIC/models/init_g1_state/pickup_table__alcohol_1__000_first_frame.pkl \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/03055Termination_walk-reference-free-book \
  '~manager_env/recorders=empty' '+manager_env/recorders=render'
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

  rsync -aP \
  --no-owner --no-group \
  --exclude='data/' \
  --exclude='imports/blender/' \
  --exclude='outputs/' \
  -e "ssh -p 9980 -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
  /home/tide/robot/GRAIL/ \
  yuguanchao@zz.irmv.top:/home/yuguanchao/data0/GRAIL/
  
    rsync -aP \
  --no-owner --no-group \
  --exclude='imports/blender/' \
  --exclude='outputs/' \
  -e "ssh -p 9980 -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
  /home/tide/robot/GRAIL/ \
  yuguanchao@zz.irmv.top:/home/yuguanchao/data0/GRAIL/

  ssh -p 9980 yuguanchao@zz.irmv.top


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
