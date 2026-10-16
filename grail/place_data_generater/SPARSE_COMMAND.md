<!-- Sparse 接入说明：运行说明文档，无代码导入；可删。逐文件用途、引用树与删除影响见 SPARSE_COMMAND_FILES.md。 -->

离线 sparse command 使用 `place_generated_start_frame=a`、
`place_release_start_frame=b`，要求 `0 <= a < b < T`。标签 `policy_command_x_yaw` 为 float32 [T,2]，内容是 dx（米）和 dyaw（弧度）。
仅 dx 前瞻 0.5 秒，截断在 `b-1`。dyaw 在同向转向段内为固定的总转角；
累计超过 0.2 rad 才确认转向段，段外为零，总角度限幅 ±1.5π。
相隔不超过 0.2 秒且偏转不超过 0.02 rad 的同向片段合并，以容忍短暂停顿和抖动。机器人四元数为 xyzw，
释放前物体目标四元数为 wxyz。

服务器数据入口：

```bash
bash grail/place_data_generater/server_concat_motion.sh
```

拼接完成后调用 `concat.py` 中的 `generate_sparse_commands` 统一生成标签。重新拼接时先去掉
源机器人中的派生标签，再按最终时间线生成，写入机器人、meta 和报告。

目标实验是 `robocasa_carry_table`。
学生 privileged 输入为 87 维：原有状态 48 维、释放目标 18 维、搬运目标 18 维、
sparse command 和 drop button 共 3 维。动作仍为 64 维完整 latent 和 2 维手部动作。
教师仍使用 `models/pnp_table/last.pt` 及自己的原始观测接口。motion loader
要求 float32 标签、正确形状和有限数值。离散标签与教师手部动作按实际目标帧
时间做零阶保持，并与机器人使用同一裁剪区间。没有标签时直接报错。

物理拾起由右手接触力达到 0.1 N、物体不再受桌面支持连续 3 个仿真步判断；
释放由按钮触发后连续 3 步失去手部接触判断。释放前连续 3 步失去手部接触
记录为掉落。状态只在仿真步中更新，reset 清除对应环境的状态。观测读取
保持纯读取；只有实际持物且按钮未触发时提供搬运命令。

释放目标观测、目标奖励和成功记录统一读取 `get_release_target_pose()`；松手后
停止物体参考位置偏差终止检查，避免参考自由落体穿地后造成错误提前终止。
学生完整 latent 不受教师 residual fade-out 缩放。

两个目标观测分别从同级 `meta/<motion_key>.pkl` 的 `place_object_goal_pos_w` /
`place_object_goal_quat_w` 和 `place_object_carry_goal_pos_w` /
`place_object_carry_goal_quat_w` 读取世界位姿（四元数为 wxyz）。搬运目标通过
`load_carry_target_pose: true` 启用，缺少字段或位姿无效时直接报错。
两者都沿用物体的 `zero_root_xy` 平移、环境原点和 `object_z_offset`，并在每次
观测时使用当前仿真的 pelvis 和手部位姿重新转换到相对坐标系。每个目标输出
pelvis 系 3D 位置 + 6D 旋转、手部系 3D 位置 + 6D 旋转，共 18 维；相对位姿不缓存。

服务器正式训练入口：

```bash
cd /home/GRAIL/imports/SONIC
/isaac-sim/python.sh gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/distill/robocasa_carry_table \
  headless=True \
  algo.config.teacher_checkpoint=models/pnp_table/last.pt \
  manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=gear_sonic/data/assets/robot_description/mjcf/
```

直接使用正式实验的 15000 iterations、学生控制和原有 PPO/BC 调度。

参考回放直接加载新 checkpoint 保存的观测配置和标签。
