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

拼接时默认只把生成动作中第一个识别到的转身段扩展为两倍帧数，例如 25 帧变为
50 帧，输出 FPS 不变，其他段保持原速。位置和关节角均匀线性插值，root 四元数
沿最短路径线性插值后归一化。物体轨迹根据插值后的右掌姿态重新计算，接触点、
保持段、释放段和 sparse command 均使用扩展后的时间线。没有识别到转身时不扩展。

通过 `TURN_SLOWDOWN_FACTOR=3 bash grail/place_data_generater/server_concat_motion.sh`
可改为三倍帧数，设置为 `1` 则保持原速；倍率必须是大于等于 1 的有限数。
直接调用 `concat.py` 时可传 `--turn-slowdown-factor`，调用 `append()` 时传
`turn_slowdown_factor`。报告的 `turn_slowdown` 和 meta 的 `place_turn_slowdown`
记录倍率、扩展前后帧数及最终时间线中的 `[start_frame,end_frame_exclusive)` 区间。

拼接完成后调用 `concat.py` 中的 `generate_sparse_commands` 统一生成标签。重新拼接时先去掉
源机器人中的派生标签，再按最终时间线生成，写入机器人、meta 和报告。

已有拼接数据可以单独重算标签，无需重新生成动作。在仓库根目录运行：

```bash
python -m grail.place_data_generater.regenerate_sparse_commands \
  --dataset-dir data/hf_dataset/data_update/data/pickup_table_place_concat \
  --output-dir data/hf_dataset/data_update/data/pickup_table_place_concat_sparse
```

输出目录必须不存在。脚本先复制完整数据集，再更新副本的 robot、meta 和 reports；
原数据保持不变。复制优先使用 reflink，不支持时普通复制，完整副本约需 30 GB。
旧的三列 `policy_command_xy_yaw` 会被替换为重新计算的两列 `policy_command_x_yaw`，
同时重算 `policy_command_drop_button` 和 sparse command 元信息。机器人与物体轨迹、
接触点、FPS、BPS、USD 和视频保持原样。后续评估把数据路径统一切换至输出目录。

直接更新原目录时改用 `--in-place`：

```bash
python -m grail.place_data_generater.regenerate_sparse_commands \
  --dataset-dir data/hf_dataset/data_update/data/pickup_table_place_concat \
  --in-place
```

这只更新原目录中的 robot 派生标签、meta 的 sparse command 元信息及释放目标、
reports 的 sparse command 统计，不复制数据集。

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
