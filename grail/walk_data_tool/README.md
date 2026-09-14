# Kimodo → GRAIL 动作拼接

将 Kimodo 行走 CSV 放在 GRAIL 抓取 motion 之前。固定抓取段 B 的世界坐标，
用 pelvis yaw 和双脚中心 XY 对齐行走段 A 的最后一帧。导出数据，不生成视频。

## 运行

在仓库根目录、已有 GRAIL Python 环境下运行（需要 NumPy、SciPy、joblib 和 SONIC FK 的依赖）：

```bash
conda activate grail
python grail/walk_data_tool/concat_walk_motion.py
```

默认读取：

- A：`data/kimodo/demo/qpos.csv`
- B：`data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot/pickup_table__alcohol_12__001.pkl`
- 同名的 `objects/`、`meta/`、`bps/`、`object_usd/` 配套文件。

默认写到 `data/hf_dataset/data_update/data/pickup_table_walk_concat/`：

```text
robot/pickup_table__alcohol_12__001.pkl
objects/pickup_table__alcohol_12__001.pkl
meta/pickup_table__alcohol_12__001.pkl
bps/pickup_table__alcohol_12__001.npy
object_usd/pickup_table__alcohol_12__001.usd
reports/pickup_table__alcohol_12__001.json
```

输入存在 BPS 的 `_*.npy` 共享文件时也会复制。BPS、USD 按原文件复制，
meta 保留桌子等原有字段，并新增 `total_frames`、`transition_start_frame`、
`transition_end_frame`。当前样例分别为 **513、243、272**，起止索引从 0 开始且两端都包含，
表示 `build_transition` 生成的完整 30 帧替换窗口（包含 A/B 被重写的帧）。
不会复制原数据集的清洗报告或其他 motion。当前样例 USD 没有外部引用。
后续若换成带外部资源的 USD，需要同时提供那些资源并保持引用路径。

可覆盖输入和输出路径：

```bash
python grail/walk_data_tool/concat_walk_motion.py \
  --walk-csv data/kimodo/demo/qpos.csv \
  --dataset-dir data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded \
  --motion-name pickup_table__alcohol_7__000 \
  --output-dir data/hf_dataset/data_update/data/pickup_table_walk_concat \
  --transition-frames 20
```

`--transition-frames N` 默认为 10，同时控制新增过渡帧数和两侧控制点距离：
10 对应 `A[-11], A[-1], B[0], B[10]`，20 对应
`A[-21], A[-1], B[0], B[20]`。A 末帧和 B 首帧之间插入 N 帧，
因此两个端点相隔 N+1 个时间步。每段输入至少需要 N+1 帧。
完整替换窗口为 3N 帧，拼接方式为 `[A_aligned[:-N], window, B[N:]]`；
输出总帧数为 `len(A) + N + len(B)`。物体、接触帧、meta 和报告索引同步调整。

已有输出默认报错；重新生成同一结果时添加 `--overwrite`。
输出目录不能是输入数据集目录、其父目录或子目录。默认路径按脚本位置解析，
传入的相对路径按当前工作目录解析。

## 时间轴和数据约定

以下以默认 `--transition-frames 10` 为例：当前输入 A 为 253 帧，B 为 250 帧、25 Hz。
加入 10 帧过渡后，输出 **513 帧、25 Hz**：

- A 占索引 `0..252`，过渡占 `253..262`，B 占 `263..512`，两段端点都保留。
- 插值采用四点 PCHIP，控制点是 A[-11]、A[-1]、B[0]、B[10]，即输出索引
  `242、252、263、273`。在输出索引 `243..272` 上重新求值，覆盖恰好 30 帧：
  A 最后 10 帧 + 新增 10 帧 + B 最前 10 帧。控制点的实际间隔为 `10、11、10` 帧。
- 根位置、关节角及手指数据逐分量 PCHIP；四元数控制点先连续符号校正，逐分量
  PCHIP 后再归一化。`pose_aa` 从插值后的根四元数和 DOF 重建。
  A 末帧、B 首帧的数据（含四元数符号）原样保留，避免改变控制点。
- 两段各需至少 11 帧。A 的 `0..242` 和 B 的 `10..249` 保持原样（A 已做 SE(2) 对齐）；
  B 的 `1..9` 允许被插值改写，不再承诺整个 B 数组不变。没有对 B 做整体平移或旋转。
  25 Hz 下仍只新增 0.4 秒，A 原末帧到 B 原首帧相隔 11 个时间步，即 0.44 秒。
- CSV 不含 FPS，本工具按 B 的 FPS 解释全部 CSV 帧，不重采样。
  如果 CSV 原本为 30 Hz，行走播放速度会变为原来的 5/6。
- CSV 为无表头的 36 列：`xyz + wxyz + 29 DOF`，米、弧度、Z-up。
  机器人 PKL 的 `root_rot` 用 **xyzw**；物体 `root_quat` 用 **wxyz**。
- 身体顺序来自 Kimodo `exports/mujoco.py` 对 G1 XML 的 joint 遍历。
  本工具存有明确的源关节名称和轴定义，与 GRAIL MJCF 按名称映射、校验旋转轴。
  CSV 应使用官方转换器默认的 `mujoco_rest_zero=False` 导出方式。
- PCHIP 前的 A 保留肩肘摆臂等身体动作，六个腕 roll/pitch/yaw 固定为 `0`。
  这是相对前臂的中立腕姿，手掌仍随手臂运动，不锁定世界朝向。
- A 的 14 个三指手 DOF、左右手标量动作分别重复 B 首帧原值。
  手指排列和标量动作编码保持原样；不由手指角度均值重新推算动作。
- B 首帧和第 10 帧及之后的所有机器人数组完整保留；重建 A 和 PCHIP 窗口的 `pose_aa`。
  `smpl_joints` 保持与源导出器一致的全零占位。
- 物体在 A 和过渡阶段保持 B 首帧的位姿；这两个阶段无手物接触，B 的接触帧索引整体加 263。
  meta 中桌子的位姿、尺寸以及 BPS 均不做坐标变换。

GRAIL 当前 `MotionLibRobot` 会在加载时做 50 Hz 重采样并从 FK 推导速度。
导出阶段增加 10 帧并用 PCHIP 重写接缝附近的 30 帧，不做全段重采样、IK 或额外高度修正。
窗口内 root Z 也做 PCHIP 插值；不保证脚接触固定或边界速度连续。
当前 29 维 `dof` 文件的训练路径使用 `hand_action_left/right`；独立的 14 维
`hand_dof_pos` 完整保存在 PKL 中供回放等消费者读取，现有 loader 不会自动把它追加到 29 维身体 DOF。

## 训练 / 评估加载

在你原有训练或评估命令后使用以下覆盖项（这里以本机路径为例，部署到 `/home/GRAIL` 时相应替换前缀）：

```bash
++manager_env.config.object_usd_path=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/object_usd \
++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/robot \
++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/objects \
++manager_env.commands.motion.motion_lib_cfg.bps_dir=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat/bps \
++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/
```

`meta/` 按 `robot/` 的同级目录自动查找。文件名和内部 key 保持原有关系。
若要从行走第一帧开始评估，额外设置
`++manager_env.commands.motion.start_from_first_frame=True`。

## 验证与扩展

脚本在临时目录完成写出、回读、B 第 10 帧起的后缀、A 保留前缀和内部控制点逐值及 dtype 比对、静态资源 SHA-256 比对、
现有 `validate_motion_input` 校验后，再发布对应文件。JSON 报告记录帧数、帧率策略、
过渡帧数与方法、拼接索引、SE(2) 旋转和平移、中心 / yaw 误差及单脚残差。

```bash
python -m unittest grail.walk_data_tool.test_concat_walk_motion -v
```

测试包括实际样例导出、CPU `MotionLibRobot` 加载、50 Hz 机器人 / 物体 / 接触 / 手动作读取。
CPU loader 仍会创建 multiprocessing 本地套接字，运行环境须允许本地进程通信。
A 原末帧与 B 原首帧的中心误差要求 `<1e-5 m`，yaw 误差要求 `<1e-5 rad`。
该样例的左右单脚水平残差各约 4.2 cm，这是站姿差异；本版本只对齐中心，不要求两脚分别重合。

脚本接口分为 CSV 适配、GRAIL FK、`compute_se2_alignment`、`transform_motion_se2`、
`build_transition`、`concat_motion`、物体扩展与导出。`concat_motion` 接受多个同格式、同帧率的 motion，
不隐式对齐；`build_transition` 默认产生 30 帧替换窗口，随后按 `[A_aligned[:-10], window, B[10:]]` 拼接。
后续批量任务可复用导出函数，IK 可放在过渡生成与拼接之间。
世界位置、xyzw 旋转、世界速度和局部关节数据有各自的处理类别；未知字段直接报错，
需新增明确的适配策略。首版 CLI 处理一组 CSV + 抓取 motion。
