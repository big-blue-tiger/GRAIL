# robocasa_carry_table 的 student observation

入口：`gear_sonic/train_agent_trl.py +exp=manager/universal_token/distill/robocasa_carry_table`。本文只列 student actor 实际读取的 observation。用户命令中的数据路径、环境数量和 loss schedule 覆盖不改变 observation 项的组成。

## 输入与阶段

配置继承链：`robocasa_carry_table` → `robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint_micro_step` → `robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint` → `robocasa_pickup_table_mlp_decoder_latent_vector_obs`。

Student 使用 `EncoderJointVectorMlpPolicy`，将以下两个组展平后直接拼接：

- `proprio_obs`：`[B, 5, 161]`，5 帧历史，展平为 `[B, 805]`。
- `privileged_obs`：`[B, 87]`，仅当前帧，无历史。
- 最终 MLP 输入：`[B, 892]`。本次只置零，不删除 observation，不改变 shape 或拼接顺序。两个组均关闭 observation corruption，且受修改的项没有额外的 scale/noise。

`B = num_envs`，用户命令中为 `9216`。下表为配置预期的 shape；其中 BPS 长度由磁盘文件实际决定，本配置要求为 10，用户给出的服务器数据不在本地，未实测其 BPS 文件。

令 `t = command.motion_start_time_steps + command.time_steps`，`S = command.motion_lib._place_generated_start_steps[command.motion_ids]`。`t` 是含随机起始偏移的已加载轨迹帧号，`S` 已与轨迹重采样和裁剪对齐，各并行环境分别判断。

- `t < S`：保留 carry goal 和桌面，release goal 的全部 18 个值置零。
- `t >= S`：保留 release goal，carry goal 的全部 18 个值和桌面的全部 12 个值置零。

边界帧 `t == S` 进入 place 阶段，与现有 `is_place_generated_phase()` 及该实验 reward 的阶段划分一致。即使从中途随机 reset，也直接按绝对帧号获得正确的置零结果。

## 完整 observation 列表

表中名称按各组实际拼接顺序排列；历史项的 shape 包含 5 帧维度。

| 组 | 具体名称 | Shape | 值的含义 | 本次是否修改、如何修改 |
| --- | --- | --- | --- | --- |
| `proprio_obs` | `joint_pos` | `[B, 5, 43]` | 43 个关节相对默认关节位置的偏移 | 否 |
| `proprio_obs` | `joint_vel` | `[B, 5, 43]` | 43 个关节相对默认关节速度的偏移 | 否 |
| `proprio_obs` | `projected_gravity` | `[B, 5, 3]` | 重力方向在机器人 anchor/pelvis 坐标系中的向量 | 否 |
| `proprio_obs` | `base_ang_vel` | `[B, 5, 3]` | 机器人根部坐标系中的根部角速度 | 否 |
| `proprio_obs` | `base_lin_vel` | `[B, 5, 3]` | 机器人根部坐标系中的根部线速度 | 否 |
| `proprio_obs` | `last_meta_action` | `[B, 5, 66]` | 上一步策略输出：64 维 latent 加左右手各 1 维 finger primitive；episode 起点清零 | 否 |
| `privileged_obs` | `object_bps` | `[B, 10]` | 当前物体的 BPS 几何编码，episode 内静态 | 否 |
| `privileged_obs` | `table_corners_b` | `[B, 12]` | 桌面四个顶角在当前模拟 pelvis 坐标系中的 xyz | **是：`t >= S` 全部置零；`t < S` 保留** |
| `privileged_obs` | `object_pos_b` | `[B, 3]` | 当前模拟物体在当前模拟 pelvis 坐标系中的位置 | 否 |
| `privileged_obs` | `object_ori_b_6d` | `[B, 6]` | 当前模拟物体相对当前模拟 pelvis 的旋转，取旋转矩阵前两列 | 否 |
| `privileged_obs` | `hand_object_transform_6d` | `[B, 9]` | 当前模拟物体在手参考坐标系中的位置 3 维及旋转 6 维 | 否 |
| `privileged_obs` | `hand_object_contact_force_magnitude` | `[B, 8]` | 8 个右手接触 link 各自与物体接触力的 L2 模长 | 否 |
| `privileged_obs` | `target_object_poses_6d` | `[B, 18]` | metadata 的 release goal 分别在当前模拟 pelvis、手坐标系中的位姿，各 9 维 | **是：`t < S` 全部置零；`t >= S` 保留** |
| `privileged_obs` | `target_object_carry_poses_6d` | `[B, 18]` | metadata 的 carry goal 分别在当前模拟 pelvis、手坐标系中的位姿，各 9 维 | **是：`t >= S` 全部置零；`t < S` 保留** |
| `privileged_obs` | `sparse_target_root_trajectory_command_contact_aware` | `[B, 2]` | 参考 heading 坐标系下的稀疏 dx（米）及转身阶段总 dyaw（弧度）命令 | 否，保留原有接触/drop 门控 |
| `privileged_obs` | `drop_button` | `[B, 1]` | 当前绝对参考帧的离线释放请求标签，释放边界前为 0，从边界开始为 1 | 否 |

`proprio_obs` 每帧为 `43 + 43 + 3 + 3 + 3 + 66 = 161` 维。

`privileged_obs` 为 `10 + 12 + 3 + 6 + 9 + 8 + 18 + 18 + 2 + 1 = 87` 维。

## 具体值的布局

- `table_corners_b`：桌子局部坐标中四个顶角按 `(-x,-y)`、`(-x,+y)`、`(+x,-y)`、`(+x,+y)` 排列，变换至当前 pelvis 系后展平为 `[x0,y0,z0,...,x3,y3,z3]`。当前桌子尺寸配置为 `[2.0, 0.6, 0.04]`；顶面局部 z 为半厚度。
- 两个 goal：`[pelvis_pos(3), pelvis_rot6d(6), hand_pos(3), hand_rot6d(6)]`。metadata 世界目标加上环境 origin 和 `object_z_offset` 后，每次 observation 均根据当前模拟 pelvis/手位姿重新计算相对位姿。
- 所有 rot6d 均由旋转矩阵前两列按行展平：`[R00,R01,R10,R11,R20,R21]`。阶段禁用时位置和旋转一起清零，包括旋转矩阵中原本为 1 的分量。
- 手参考坐标系的位置由 `object_to_hand_frame_transformer` 所配置的目标 link 位置取均值，方向取第一个目标 link 的四元数。
- 8 个接触力模长的顺序：`right_hand_palm_link`、`right_hand_thumb_0_link`、`right_hand_thumb_1_link`、`right_hand_thumb_2_link`、`right_hand_index_0_link`、`right_hand_index_1_link`、`right_hand_middle_0_link`、`right_hand_middle_1_link`。
- 稀疏命令保留现有行为：确认拿起前、失去物理手接触时、或 `drop_button >= 0.5` 时置零，不额外按 generated 阶段置零。`drop_button` 表示参考释放请求，不代表物理上已经松手。

该 student 的桌面 observation 只有 `table_corners_b`；`table_pos_b`、`table_ori_b`、`table_geometry` 不在其输入中。也不读取 camera、`actor_obs`、`critic_obs`、`policy_atm` 或 tokenizer observation。新增 `phase` 参数只在该实验的 student 三个 term 中启用，默认 `None` 保留原行为。

## 无参考推理中的动态 S

无参考推理保留这三个 observation 的 `phase` 参数，并由 `ReferenceFreeCommand.is_place_generated_phase()` 提供在线阶段判断。它与 sparse 序列播放共用 `OfflineCarryTask.has_playback_started()`，不需要 motion library。

每个环境首次确认拿起物体时，设置 `S = task.start_step = 当前控制步 + settle_steps`；`settle_steps` 由 `pickup_settle_s / step_dt` 向上取整得到，默认等待 1 秒。在线 place 阶段条件为 `(S >= 0) & (task.steps >= S)`：

- 尚未确认拿起，或拿起后仍在等待：release goal 全零，carry goal 和桌面保留。
- 到达 `S`、开始播放 sparse 第 0 帧的同一个 observation：release goal 保留，carry goal 和桌面全零。
- 此后即使命令数值为零、失去手接触、物体掉落或进入释放阶段，也不会切回之前的阶段。
- 对应环境 reset 时，`start_step` 重置为 `-1`，恢复拿起阶段；其他并行环境不受影响。

这里依据的是播放时钟的启动条件，不是 sparse command 首次非零的数值检测。读取 observation 和阶段判断均不推进时钟或修改状态。
