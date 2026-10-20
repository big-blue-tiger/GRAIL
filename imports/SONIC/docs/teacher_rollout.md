# Teacher rollout 导出

在现有 `gear_sonic/eval_agent_trl.py` play 命令末尾追加：

```bash
++teacher_rollout_dir=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_teacher_rollout
```

开启后会设置 `run_once=True`、`start_from_first_frame=True` 和
`sample_before_contact=False`，并关闭随机起点、将 `eval_start_frame` 置零，
每个环境仅保存首个 episode。
原命令的 `num_envs=4` 会导出当前分片选中的 4 条 motion；同名文件再次运行时覆盖。

输出沿用 `pickup_table_place_left_right` 的参考数据格式：

- `robot/<motion>.pkl`：保留源文件内部 motion key、字段、dtype 和 FPS（当前数据为 25）。
  `dof`、`root_trans_offset`、`root_rot`、`pose_aa`、`hand_dof_pos` 来自实际仿真状态，
  `hand_action_left/right` 来自实际执行的开合手指令。
- `objects/<motion>.pkl`：直接记录仿真刚体 actor frame 的实时位姿，
  `root_pos [T,1,3]` 为去掉环境 origin 后的位置，`root_quat [T,1,4]` 为 wxyz。
  与 `robot/` 使用完全相同的采样帧、帧数和 FPS。接触点仍为源数据的物体局部表面标签。
- `meta/`、`reports/`：沿用源元数据，更新帧数；提前结束时裁剪阶段帧索引。
  `place_object_goal_*` 更新为 rollout 释放前一帧的物体位姿，
  `place_object_carry_goal_*` 更新为 rollout 对应 carry source 帧的物体位姿。
- `bps/`、`object_usd/`：复制对应 BPS、USD 及纹理。

记录发生在 action 执行后、自动 reset 前，包含终止状态。
从 50 Hz 仿真结果中保留每个源数据帧时间区间末尾的状态；提前终止时也保存最后一个不足整帧的区间。
所有位置减去对应环境的 origin，机器人四元数为 xyzw，物体四元数为 wxyz，
身体关节为 MuJoCo 顺序，手指关节为源数据的 14 DOF 顺序。

`policy_command_x_yaw`、`policy_command_drop_button`、SMPL 零占位和接触标签沿用源数据，
按导出帧数截取。提前终止的 episode 也会保存，不按成功与否筛选。
视频仍由原命令的 `render_envs.video_save_path` 单独保存，导出不依赖渲染是否开启。
带物体的 play 默认在每个环境画两个半径 5 cm 的小球：绿色标注释放 goal，蓝色标注 carry goal。
位置分别读取 `get_release_target_pose()` 与 `get_carry_target_pose()`，即 meta 中的
`place_object_goal_pos_w`、`place_object_carry_goal_pos_w` 加环境 origin。
每次渲染前更新，因此 GUI 和 `headless=True` 的录制视频都能显示；不额外抬高目标点，
也不会自动开启机器人参考骨架可视化。`++show_object_goal=False` 可关闭小球。
缺失的历史 `objects/` 不能由机器人轨迹或原参考物体轨迹补回，需带导出参数重新运行 teacher。

## 随机帧初始化测试

在 play 命令末尾追加：

```bash
++manager_env.commands.motion.eval_random_start=True
```

当前临时改为在每个 motion 的末尾 10 秒内均匀采样初始帧；不足 10 秒时采样全段，覆盖固定首帧、
`eval_start_frame`、前 N 帧和抓握前采样设置；reset 后会重新采样。
机器人身体、物体和手指姿态使用同一时间点，手指读取保存的 `hand_dof_pos`。
默认仍沿用原来的固定帧评估行为。修改 `seed=123` 可更换采样结果。

`pnp_table` play 与 `robocasa_carry_table` 训练统一使用 `init_z_offset=0.05`，
`object_init_z_offset` 引用同一值：每次 reset 时机器人和物体都额外抬高 5 cm。
`object_z_offset=0`，因此桌子、目标点与参考轨迹坐标保持原值；该偏置只作用于写入仿真的初始位置。

测试刚生成的 teacher rollout 时，将原命令的 `object_usd_path`、`motion_file`、
`object_motion_file` 和 `bps_dir` 四个路径中的 `pickup_table_place_concat` 改为
`pickup_table_teacher_rollout`，并追加上面的开关。测试时不添加 `teacher_rollout_dir`。
保留 `run_once=True` 时，4 个环境各测试一次随机起点；改为 `run_once=False` 可持续采样。
