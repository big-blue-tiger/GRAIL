# 无 reference 的 carry-table student play

从 `imports/SONIC` 目录使用 Isaac Lab Python 环境运行。使用
`robocasa_carry_table` 训练出的 **87 维 privileged 输入** checkpoint，入口仍为
`gear_sonic/eval_agent_trl.py`。下面保留 book 新场景、8 环境和视频录制：

```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/1016robocasa_carry_table-20261016_153147/last.pt \
  +headless=True ++num_envs=8 +run_once=True \
  ++manager_env.config.episode_length_s=30 \
  reference_free.enabled=True \
  reference_free.terminate_on_success=False \
  reference_free.dataset_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_left_right \
  reference_free.initial_pose_file=/home/GRAIL/imports/SONIC/models/init_g1_state/pickup_table__alcohol_1__000_first_frame.pkl \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/1016robocasa_carry_table-reference-free-book \
  '~manager_env/recorders=empty' '+manager_env/recorders=render'
```

不需要设置 `motion_file`、`object_motion_file`、motion library 的资产路径或
tracking 终止条件。场景目录只需配对的 `object_usd/`、`bps/`、`meta/` 和
`objects/`；物体轨迹文件仅用于提取首帧位置和方向。运行时不加载机器人参考轨迹，
也不读取 teacher 数据、预存 sparse 标签或服务器。

## 双目标与指令

每个环境保留自己的物体，reset 重新采样机器人、物体 XY、双目标及离线指令：

- carry 保留物体初始 XY 和方向，抬升 0.20–0.35 m。
- release 在桌外 ±75° 扇区，距 carry 的水平距离按面积均匀采样于 2.5–3.2 m，
  比 carry 降低 0.05–0.14 m；方向施加计划转角。转向取初始航向到采样最终航向的
  最短有符号角差 `atan2(sin(final_yaw-initial_yaw), cos(final_yaw-initial_yaw))`，
  正值左转、负值右转，不再按环境或 episode 强制交替；绝对角差不超过 0.2 rad 时不转向。
- 按固定持物偏移生成短弧线和直线，再由同一路径计算 release 目标和指令。
  转向角速度为 1.3 rad/s，转向时前进 0.15 m/s，直行速度为 0.35 m/s。

首次确认物理拾起后等待 1 秒，开始播放预先生成的 **25 Hz** 序列：

| 阶段 | 指令 |
| --- | --- |
| 等待拾起、稳定 | dx、dyaw、drop 为零 |
| 弧线转向 | dx 从路径提取，dyaw 始终为该段总转角 |
| 直线搬运 | dx 从路径提取，dyaw 为零 |
| 保持 0.72 秒 | dx、dyaw、drop 为零 |
| 松手至 episode 结束 | dx、dyaw 为零，drop 为 1 |

dx 是前瞻 12 帧的航向系位移，单位为米，沿用 0.02 m 死区和 0.15 m 上限；
dyaw 单位为弧度。仿真按时间零阶保持读取，时间轴不随位置误差或接触变化暂停，
松手按计划发出。提前掉落锁存失败并屏蔽搬运指令，不重新启动。
接触门控复用训练代码：0.1 N 阈值、连续 3 个控制步确认拾起、释放或提前掉落。

参数集中在 `gear_sonic/config/base_eval.yaml` 的 `reference_free.planner` 下，
例如 `reference_free.planner.pickup_settle_s=1.5`、
`reference_free.planner.walk_speed=0.3`。采样范围来自对
`pickup_table_teacher_300` 全部 300 条样本的离线统计，无运行时依赖。

## 模型与结果

观测保持 checkpoint 的顺序：48 维状态 + release 18 + carry 18 + sparse 2 +
drop 1，共 87 维；动作仍为 64 维 latent + 2 维手部动作。两组目标复用训练的
观测函数，每次根据当前 pelvis、手部位姿计算相对位置和 6D 旋转。
机器人 root 的 xyzw 转为 wxyz；物体和桌子的 wxyz 沿用训练约定。
机器人保留初始 roll/pitch 并面向桌子。机器人和物体 reset 高度补偿沿用 checkpoint，
通常均为 0.05 m；桌子和目标不加该补偿，重复 reset 不累加高度。

蓝色球为 carry、绿色球为 release；线框表示初始化采样区域。
`reference_free.debug_vis=False` 关闭标记。

成功要求先持物进入 release 目标的 `reference_free.success_threshold=0.1` m
范围，再在该范围内确认物理释放；播放完成或只到达目标均不直接算成功。
成功后自然下落不撤销结果。`reference_free.terminate_on_success=False` 时继续至超时，
否则成功后结束 episode。`run_once=True` 仅统计各环境首个 episode。

`reference_free_results.json` 写入渲染目录，可通过 `+run_once_report_path=...` 覆盖。
报告包含双目标、最终位置、拾起/到达/松手请求/物理释放/提前掉落及其时间，
以及计划阶段边界。位置包含环境 origin；`final_distance` 是诊断值，松手下落后
它可能增大，而已锁存的 `success` 仍为真。视频继续使用现有 render recorder。

旧 **66 维 privileged 输入**的 joint_micro_step checkpoint 自动沿用原来的单目标
抬升测试：目标在初始位置上方 0.4–0.7 m，成功仍按终止时最终位置判断。

## CPU 检查

在仓库根目录运行：

```bash
python -m unittest discover -s tests -p test_reference_free.py
python tests/check_reference_free_runtime.py
```

第一项对照训练生成器验证几何、标签、时钟、接触门控和结果锁存；第二项使用
最小场景替身检查新旧模式各 8 场景的 reset、补偿及终止记录，不执行真实物理仿真。
完整 8 环境视频与物理效果仍需在安装了 Isaac Lab 的环境中运行上面的 play 命令验证。
