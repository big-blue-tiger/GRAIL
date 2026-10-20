# Place 拼接数据的 meta / report 字段与运行时用途

本说明按 2026-10-20 当前工作区代码核对，覆盖用户给出的三个入口：

| 记号 | 入口 | 本文所指行为 |
| --- | --- | --- |
| 训 | `train_agent_trl.py +exp=manager/universal_token/distill/robocasa_carry_table` | 学生 PPO/BC 蒸馏，包括教师动作、教师目标、观测、奖励、终止与记录器 |
| 推 | `eval_agent_trl.py reference_free.enabled=True` | 无参考学生推理；提前进入 `reference_free_eval.run()`，不创建原来的 motion library |
| 师 | `eval_agent_trl.py +exp=manager/universal_token/hoi/pnp_table ++teacher_rollout_dir=...` | 教师参考回放与实际仿真轨迹导出 |

这里的 meta 是数据集同级 `meta/<文件名>.pkl` 的字典，不是训练日志目录里的 `meta.yaml`。robot / objects PKL 外层的内部 motion key 可以与文件名不同；本文 `name` 指文件名主干。

用户训练路径 `pickup_table_teacher_300`、推理数据 `SBTO/pickup_book/pickup_table` 和服务器 checkpoint 不在本地，无法断言它们实际保存的数值及 checkpoint 的观测维度。下文严格区分当前源码公式、本地已有数据和需由实际文件确认的配置。没有启动训练或 Isaac Sim。

`eval_agent_trl.py` 还会先加载 checkpoint 旁（或上一级）的 `config.yaml` 与其 `eval_overrides`，再合并命令行配置。下表“当前 pnp 未启用”的判断针对本地 pnp 实验声明及默认开关；若服务器 checkpoint 保存的配置额外打开了某项，最终应以合并配置为准。`reference_free` 分支会明确替换旧 motion command，因而该分支不读 place 目标/阶段字段的结论不受这些旧开关影响。

## 1. 首先明确时间轴、坐标和符号

所有帧号从 0 开始，默认是最终输出 PKL 在其 FPS 下的帧索引；名字含 `source` 的字段记录 place 拼接输入的帧索引/数量。该输入通常已经是 walk + pickup 的拼接结果，不一定是最初原始 pickup 文件。

| 符号 | 定义 |
| --- | --- |
| `F` | `float(robot["fps"])`，单位 Hz；Kimodo 原生成 FPS 为 30，输出仍使用源 robot FPS |
| `S` | place 输入完整帧数 `source_frames` |
| `C` | 源右手非空接触标注的帧号集合；`N_C=len(C)`，`D=N_C/F` |
| `c` | `record["carry_source_frame"] = max(0, S-1-round(N_C/2))` |
| `l` | `record["lower_source_frame"] = max(0, S-1-round(N_C*r))`，当前源码 `r ~ Uniform(0.6,0.8)` |
| `n` | 保留输入 `[0,c]`，故 `n=c+1` |
| `J` | 新插入 seam 帧数，`max(1,round(0.1*F))` |
| `G` | 经 30 Hz → `F` 重采样、首个转身段减速之后的生成帧数 |
| `H` | 放手前静止持物帧数，`max(1,round(0.7*F))` |
| `R` | 最后开手/物体下落帧数，`max(1,round(0.7*F))` |
| `a` | 首个生成帧 `n+J` |
| `h` | 放手前 hold 起点 `a+G` |
| `b` | release / final hold 起点 `h+H` |
| `T` | 输出总帧数 `b+R = n+J+G+H+R` |
| `g` | walk 输入 report 的 `b_start_index`，当前写为 `grail_start_frame` |
| `p_o(t), Q_o(t)` | 物体的世界位置和旋转矩阵；位置米、世界 Z 向上 |
| `p_h(t), Q_h(t)` | FK 计算的右掌世界位置和旋转矩阵 |
| `p_r(t), ψ(t)` | robot 根部位置与绕世界 Z 的 yaw，yaw 单位 rad |

时间轴为：

```text
保留源段       seam             生成段          持物 hold        开手/下落
[0,n)          [n,a)            [a,h)           [h,b)            [b,T)
末帧 c=n-1     共 J 帧          共 G 帧         共 H 帧          共 R 帧
                残差渐出从 c 到 a，包含两个端点
```

区分帧号与帧数：`end_frame` 通常包含端点，`end_frame_exclusive` 不包含端点，`total_frames=T` 是数量；末帧是 `T-1`。帧时间戳为 `i/F`，`T` 个采样点首尾时间差为 `(T-1)/F`。

robot `root_rot` 是 **xyzw**；Kimodo CSV 根四元数、物体 `root_quat`、目标四元数是 **wxyz**。本说明涉及的目标是数据集世界位姿，尚未加并行环境的 origin。桌子四元数在当前运行代码中也按 wxyz 消费，不能因某份数据写了 `[0,0,0,1]` 就自动解释为单位旋转。

## 2. 统一 meta 字段列表

`meta = deepcopy(joblib.load(源meta))` 后才 `update(...)`，因此下面的显式写入字段不是完整封闭 schema：任何源 meta 旧字段都会继续保留。用途列中“导出裁剪”指教师保存时的通用帧字段裁剪，不表示控制器使用该字段。

### 2.1 当前代码写入的时间边界与来源字段

| meta key | 物理含义、单位与计算公式 | 训 | 推 | 师 |
| --- | --- | --- | --- | --- |
| `total_frames` | 输出采样点数量 `T=len(robot_out["dof"])` | 残差渐出合法性检查；BC carry 边界合法性检查。运动实际长度仍由轨迹数组确定 | 不读 | 渐出合法性检查；保存时改为实际导出帧数 `T'` |
| `grail_start_frame` | 源 walk report 的 `b_start_index=g`；代表旧 GRAIL pickup 段的名义起点，可能位于旧 walk→pickup 混合窗内 | 与 `a` 定义身体 tracking 的关闭区间 `[g,a)` | 不读 | 当前 pnp 配置未启用这组 tracking gate；导出裁剪 |
| `residual_fade_out_start_frame` | `n-1=c`；从保留的最后 carry 帧开始关闭教师 latent residual | 实际参与教师动作和蒸馏目标缩放 | 不读 | 实际参与教师执行 residual 缩放；导出裁剪 |
| `residual_fade_out_end_frame` | `generated_start=a`；第一生成帧 residual 系数归零 | 同上 | 不读 | 同上 |
| `place_transition_start_frame` | `n`，新增 seam 第一帧 | 无直接运行读取 | 不读 | 导出裁剪 |
| `place_transition_end_frame` | `a-1`，新增 seam 最后一帧，含端点；若 `J=1` 可等于 start | 无直接运行读取 | 不读 | 导出裁剪 |
| `place_generated_start_frame` | `a=n+J`；生成运动第一帧，也是训练从 carry 目标切换到 release 目标的边界 | 实际参与奖励阶段、tracking gate、接近速度惩罚开关和生成段终止阈值 | 不读 | 当前 pnp 未启用生成阶段奖励/阈值开关；导出裁剪 |
| `place_hold_start_frame` | `h=a+G`；生成运动结束后保持身体和物体不动的第一帧 | 无直接运行读取 | 不读 | 导出裁剪 |
| `place_final_hold_start_frame` | `b=h+H`；身体继续不动、右手开始打开、物体开始自由下落 | 无直接运行读取 | 不读 | 导出裁剪 |
| `place_release_start_frame` | 同样为 `b`。生成 sparse 标签时 `drop[b:]=1`，目标取 `b-1` | **无直接按此 key 查询释放阶段**；运行读取 robot 的 drop 数组 | 不读 | 保存时直接读取，重算 release goal；也参与导出裁剪 |
| `place_source_frames` | 未裁剪 place 输入总帧数 `S` | 无直接运行读取 | 不读 | 继承保留，不随提前结束重算 |
| `place_retained_source_frames` | 源段保留帧数 `n=c+1` | 无直接运行读取 | 不读 | 继承保留 |
| `place_trimmed_source_frames` | 从源尾部删除的帧数 `S-n` | 无直接运行读取 | 不读 | 继承保留 |
| `place_carry_source_frame` | `c`，源 carry 姿态索引；同时恰好是输出保留前缀的最后一帧 | **直接读取**：限定教师终止条件导致的 BC loss 屏蔽只发生在 carry 之前 | 不读 | **直接读取**：重算 carry goal 时使用 `min(c,T'-1)`；本字段本身不裁剪 |
| `place_lower_source_frame` | `l`，用于构造终点下放姿态的源帧索引；不是输出运动的下放开始帧 | 无直接运行读取 | 不读 | 继承保留，不裁剪 |

### 2.2 目标位姿与接触统计

| meta key | 物理含义与计算公式 | 训 | 推 | 师 |
| --- | --- | --- | --- | --- |
| `place_object_goal_pos_w` | 松手前物体位置 `[3]`，`objects_out["root_pos"][b-1,0]`，m；sparse 生成函数会以相同值再次覆盖 | release 目标观测、目标距离/成功奖励、成功记录器 | 不读，目标由新规划器产生 | motion library 直接读取；用于目标显示/记录；导出改成实际 rollout 的对应物体位置 |
| `place_object_goal_quat_w` | 松手前物体姿态 `[4]` wxyz，`objects_out["root_quat"][b-1,0]`；由 `sparse_meta` **补写**，原 `meta.update` 中没有这行 | release 目标的姿态观测；上述位置奖励不检查姿态误差 | 不读 | motion library 直接读取并验证；保存时改成实际姿态 |
| `place_object_carry_goal_pos_w` | 输入源物体 `source_objects["root_pos"][c,0]`，m；是提起/持物姿态的固定目标 | carry 目标观测；生成段之前的目标奖励 | 不读 | 默认目标显示令 `load_carry_target_pose=True`，直接读取；导出改成实际 rollout 位置 |
| `place_object_carry_goal_quat_w` | 输入源物体 `source_objects["root_quat"][c,0]`，wxyz | carry 目标姿态观测，位置奖励不检查姿态误差 | 不读 | 同上，保存时改成实际姿态 |
| `place_output_contact_start_frame` | 输出右手非空接触标注集合 `C_out` 的 `min(C_out)` | 无直接运行读取；首接触由 objects 接触标签另行计算 | 不读 | 导出裁剪；不重新检测物理接触 |
| `place_output_contact_end_frame` | `max(C_out)`，含端点；因为 seam/生成/hold 都复制 carry 接触标签，当前拼接正常为 `b-1` | 无直接运行读取 | 不读 | 导出裁剪 |
| `place_output_contact_start_s` | `min(C_out)/F`，s | 无直接运行读取 | 不读 | 原样保留，不重新计算 |
| `place_output_contact_frames` | `len(C_out)`，有标注的采样帧数 | 无直接运行读取 | 不读 | 原样保留，不重新计算 |
| `place_output_contact_duration_s` | `len(C_out)/F`，s；不是 `(end-start)/F` | 无直接运行读取 | 不读 | 原样保留，不重新计算 |
| `place_source_contact_start_frame` | `min(C)`，完整源输入中的接触起点 | 无直接运行读取 | 不读 | 原样保留，含 `source` 不裁剪 |
| `place_source_contact_end_frame` | `max(C)`，完整源输入接触末帧，含端点，可能已被 place 裁掉 | 无直接运行读取 | 不读 | 原样保留 |
| `place_source_contact_duration_s` | `D=len(C)/F`，完整源接触标注累计时长 | 无直接运行读取 | 不读 | 原样保留 |

这些接触统计衡量**数据标注**，不表示接触力传感器在仿真中实际检测到的抓握时刻。源接触集合可以有空洞，所以不能用首尾跨度代替接触帧数。

### 2.3 `place_turn_slowdown` 的所有子项

当前 `concat.py` 会写入此字典，同时 report 的 `turn_slowdown` 保存相同内容。它们用于生成质量审计，没有训练/推理阶段的直接读取；teacher 保存时也不会递归修改此嵌套字典。

`slow_first_turn` 先找重采样后生成运动的第一个有效 turn stage `[s,e)`。设 `k=e-s`、减速倍率为 `v`，扩展为 `k'=round(k*v)` 帧，替换原段，FPS 不变。位置/关节线性插值；root 四元数先统一符号，再归一化线性插值。未检出转身或 `v=1` 时不扩展。

| 子 key | 具体意义/公式 |
| --- | --- |
| `factor` | 用户输入 `turn_slowdown_factor`，当前默认 `2.0`，要求有限且 `>=1`；不是改变 FPS |
| `start_frame` | 有扩展时为 `a+s`，最终输出时间轴上的减速段起点；无扩展为 `None` |
| `end_frame_exclusive` | 有扩展时为 `a+s+k'`，不含端点；无扩展为 `None` |
| `original_frames` | `k`；无扩展为 `0` |
| `expanded_frames` | `k'`；无扩展为 `0` |
| `added_frames` | `k'-k`；已经计入 `G` 和 `T`，不得再次加到总帧数；无扩展为 `0` |

### 2.4 `sparse_command_parameters` 的所有子项

这是生成参数的记录，**三个入口都不靠这个字典重建离线 sparse 标签**。训练/teacher 的 motion library 直接读 robot 的 `policy_command_x_yaw` 和 `policy_command_drop_button`。

| 子 key | 当前写入值、公式与物理含义 |
| --- | --- |
| `lookahead_s` | 默认 `0.5` s，仅平移命令向前看，yaw 使用整个转身阶段角度 |
| `horizon_frames` | `K=max(1,round(lookahead_s*F))`；Python `round` 在正好半整数时取偶数，因此 `round(0.5*25)=12` |
| `xy_deadzone_m` | `0.02` m；XY 位移范数严格小于此值时置零 |
| `xy_limit_m` | `0.15` m；对完整 XY 向量作范数上限裁剪，再选 x 分量 |
| `command_channels` | `["dx","dyaw"]`，两个通道的语义 |
| `yaw_mode` | `"turn_stage_total"`，同一转身阶段各帧填写该阶段总转角 |
| `yaw_turn_threshold_rad` | `0.2` rad；阶段总转角绝对值必须严格大于它才产生 yaw 指令 |
| `yaw_jitter_rad` | `0.02` rad；尝试跨短暂停顿合并同方向 turn runs 时允许的偏差 |
| `yaw_pause_s` | `0.2` s；合并同方向 runs 时允许的间隔帧数为 `round(0.2*F)` |
| `yaw_limit_rad` | `1.5*pi` rad；最终 yaw 命令裁剪到正负此范围 |
| `root_quaternion_order` | `"xyzw"`，生成标签时对 robot root quaternion 的解释 |
| `goal_frame` | `b-1`；目标物体姿态采样索引，teacher 导出不会更新此嵌套字段 |

设 `t∈[a,b)`，`j=min(t+K,b-1)`。平移标签的精确公式：

```text
Δp_xy = p_r(j).xy - p_r(t).xy
u = [[ cosψ(t), sinψ(t)], [-sinψ(t), cosψ(t)]] Δp_xy
L = ||u||₂
若 L < 0.02，则 u = 0
u ← u * min(1, 0.15/max(L,1e-12))
dx(t) = u.x
```

`ψ=unwrap(atan2(R[1,0],R[0,0]))`。阶段 `[s,e)` 的 yaw 命令为 `clip(ψ(e)-ψ(s),-1.5π,1.5π)`；它是总转角，不是剩余角度，也不是角速度。识别阶段时对 `|diff(yaw)|<=1e-6` 当作零增量，再按短停顿/偏差条件合并同方向 runs。

最终 robot 新增 `policy_command_x_yaw: float32[T,2]`（米、弧度）与 `policy_command_drop_button: float32[T,1]`。区间 `[a,b)` 外 x/yaw 全零；drop 在 `[0,b)` 为 0、`[b,T)` 为 1。**没有除以秒数，因此 dx 不是 m/s**。`policy_command_xy_yaw` 只在输入旧标签清理集合中出现，当前不会输出三通道版本。

### 2.5 继承字段与本地已有数据的额外字段

| key | 来源、含义与实际用途 |
| --- | --- |
| `table_pos` | 源 meta 继承的桌体世界位置 `[3]` m。训/师在 `TrackingCommand._load_table_meta()` 直接读取，reset 时加 env origin 和可选 table offset 设置桌子位置；推在 `load_scene_inputs()` 直接读取 |
| `table_quat` | 源 meta 继承的桌体姿态 `[4]`。训/师 reset 时直接交给 Isaac；推也用于摆放桌子和采样桌外方向，当前消费约定 wxyz |
| `table_size` | 源 meta 的桌体尺寸 `[宽,深,厚]` m。训/师读入缓存，缺失默认 `[1.0,0.6,0.04]`；桌体创建尺寸另受 scene config 优先级影响。推必须读取，并用于桌体尺寸/桌角观测/规划采样 |
| `object_name` | 本地数据继承值如 `"object"`；上述三入口未见按此 key 选择物体的运行逻辑 |
| `transition_start_frame` | 旧 walk→pickup 的 residual 渐入起点。上游 `concat_walk_motion.py` 为 `frames_a-old_transition_frames`；训/师实际读取。与 `place_transition_start_frame` 是两个不同接缝 |
| `transition_end_frame` | 旧 residual 渐入终点，含端点。上游为 `transition_start_frame+len(old_transition)-1`；训/师实际读取 |
| `source_start_frame` | 当前上游 walk 拼接器写入：从最初 pickup 片段开始截取的源帧号 `source_frame`。不是 place 的 carry frame；只作来源记录 |
| `source_total_frames` | 当前上游 walk 拼接器写入：最初 pickup 文件的 `original_frames`；只作来源记录 |
| `grail_end_frame_exclusive` | 本地旧产物额外存在，样例值 `311=n`；当前 place 代码没有显式写它。训的 tracking gate 实际用 `place_generated_start_frame`，不读取这个旧 end；师导出按通用 exclusive 规则裁剪 |
| `place_turn_direction` | 本地旧产物额外存在，如 `-1`。**当前 place 拼接器与其 constraints record 没有显式写入此 key**。teacher 导出却直接访问它来生成 `rollout.reference_turn_direction`，缺少时会报 KeyError。消费规则为 `value>0.02 → left`、`value<-0.02 → right`、其余 straight；不能仅凭消费者声称生产公式一定是 `sign(总yaw)` |
| 其他源 meta key | 由 deepcopy 保留。`scene_scale` 在显式加载 `motion_meta_info_path`/单 USD 对应 meta 的场景创建分支中会被读取；用户给出的目录式 USD 默认路径不会自动逐条用这个 key 建场景。不能把任意继承字段都认定为模型输入 |

本地核验：`pickup_table_place_concat` 的 2304 份 meta 使用同一套 **36 个顶层 key**；`pickup_table_teacher_rollout` 的 4 份 meta 也为同一套 36 key。它们都有 `grail_end_frame_exclusive`、`place_turn_direction`，没有当前新代码写入的 `place_turn_slowdown`；也没有当前上游可能写入的 `source_start_frame/source_total_frames`。这是已有文件与当前生成器版本的差异。

## 3. report 每一项的含义与公式

当前 place 拼接 report 是重新构造的字典，不是源 walk report 的完整拷贝；teacher 导出再读取并修改它。训和推不读取 place `reports/<name>.json`；师在执行策略时不拿这些数值控制动作，但保存数据时会读取/更新 report。

### 3.1 标识、帧数、接触和几何质量

| report key | 物理含义 / 当前生成公式 |
| --- | --- |
| `name` | 输出文件名主干，如 `pickup_table__alcohol_10__000` |
| `source_frames` | place 输入完整帧数 `S` |
| `grail_start_frame` | `g`，同名 meta 值 |
| `retained_source_frames` | `n` |
| `trimmed_source_frames` | `S-n` |
| `transition_frames` | 新增 place seam 帧数 `J`；不是整个旧 walk→pickup 的混合窗长度 |
| `turn_slowdown` | 同 `meta.place_turn_slowdown`，完整子项见 2.3 |
| `generated_frames` | `G=len(generated["dof"])`，已经包含减速增加的帧数 |
| `hold_frames` | `H=len(hold["dof"])`，放手前 hold 帧数 |
| `final_hold_frames` | `R`，开手和物体下落帧数；“hold”指身体保持，不是物体保持 |
| `total_frames` | `T=len(robot_out["dof"])` |
| `fps` | `F`，输出轨迹 FPS |
| `residual_fade_out_start_frame` | `c`，从 meta 复制 |
| `residual_fade_out_end_frame` | `a`，从 meta 复制 |
| `target_distance_m` | `record["distance_m"]`，生成约束要求的根部 XY 目标距离。当前 `radius=sqrt(Uniform(1.7²,3.3²))`，不是旧注释所写的 2–3.5 m |
| `source_contact_start_frame` | `min(C)` |
| `source_contact_end_frame` | `max(C)`，含端点 |
| `source_contact_start_s` | `min(C)/F` |
| `source_contact_frames` | `len(C)` |
| `source_contact_duration_s` | `len(C)/F=D` |
| `output_contact_start_frame` | `min(C_out)` |
| `output_contact_end_frame` | `max(C_out)`，当前正常为 `b-1` |
| `output_contact_start_s` | `min(C_out)/F` |
| `output_contact_frames` | `len(C_out)` |
| `output_contact_duration_s` | `len(C_out)/F` |
| `release_start_frame` | `b`；report 没有 `place_` 前缀 |
| `carry_source_frame` | `c` |
| `carry_pose_lookback_s` | `D/2`，从源末端回看的名义时长。实际帧间距是 `round(N_C/2)/F`，还可能被 `max(0,...)` 截断 |
| `lower_source_frame` | `l` |
| `lower_pose_lookback_s` | `D*r`，`r~Uniform(0.6,0.8)`。实际离散帧间距按 `round(N_C*r)`，可能截断；不是输出的下放持续时长 |
| `join_hand_gap_m` | `||p_h(gen_first)-p_h(source_carry)||₂`，右掌 FK 的三维位置接缝误差；测量插值 seam 之前两端差异，不是物体接触间隙 |
| `motion_key` | `str(motion_key)`，源 robot PKL 外层内部 key；与 `name` 不要求相同 |
| `kinematics_version` | 当前 `KINEMATICS_VERSION`，如 `g1_xml_joint_zero_v2`；运动学适配器版本，非物理量 |
| `root_join_gap_m` | `||world_qpos[0,:3]-carry_values[0,:3]||₂`，根部三维接缝误差 |
| `quaternion_orders` | 三个子项：`source_robot="xyzw"`、`kimodo_csv="wxyz"`、`object="wxyz"`；用于解释文件，不执行转换 |
| `actual_root_distance_m` | `||world_qpos[-1,:2]-carry_values[0,:2]||₂`，生成终点与源 carry 根部的**净 XY 位移**；不是路径累计长度 |
| `actual_hand_lower_m` | `carry_hand[2]-hand[-1,2]`，源 carry 到生成末帧右掌高度降低量；正值下降，负值上升，不是物体自由落体距离 |
| `source_robot_sha256` | manifest 记录的源 robot 文件 SHA256。拼接前还会校验源 robot/object 哈希，但 report 这里只记录 robot 哈希 |
| `kimodo_csv_sha256` | `file_hash(csv)`，生成输入 CSV 的 SHA256 |

注意生成器重采样：若输入 CSV 为 `M` 帧，则重采样后数量为 `ceil(((M-1)/30)*F)+1`，最后目标时间裁到 `(M-1)/30`；然后首个转身扩展 `added_frames`，得到 `G`。不能直接把 CSV 帧数当作 `generated_frames`。

### 3.2 report 中的物体、手部和接触策略说明

| report key | 对应的真实实现 |
| --- | --- |
| `object_orientation_policy` | 保持 carry 帧物体相对右掌的旋转：`ΔQ(t)=Q_h(t)Q_h(c)^T`，`Q_o(t)=ΔQ(t)Q_o(c)`。最后 release 段保持物体旋转不变 |
| `object_orientation_reference` | FK 掌节点 `right_hand_roll_skel`，是 `right_wrist_yaw_skel` 的固定子节点；不是仅取 root yaw |
| `object_translation_policy` | 保持 carry 抓握偏移：`p_o(t)=p_h(t)+ΔQ(t)(p_o(c)-p_h(c))`。最后 release 段，`k=1..R`：`p_o(b+k-1)=p_o(b-1)-[0,0,0.5*9.81*(k/F)^2]`。无地面碰撞求解/落地截断 |
| `contact_points_coordinate_frame` | `"object_local"`，接触点保存为物体局部坐标，不是世界坐标 |
| `contact_points_policy` | 取源 `contact_points_right_hand[c]`，复制到 seam、generated 和持物 hold；新增左手接触为空，release 段双手接触为空；源前缀保留原标签 |
| `right_hand_hold_action` | `1`，生成/seam/持物 hold 的右手闭合指令；不是手指角度 1 rad、也不是 1 N |
| `grasp_orientation_policy` | `"generated palm rotation; no local IK correction"`，按生成的掌朝向驱动物体，不做局部 IK 修正 |
| `right_hand_opening_policy` | release 段 `hand_action_right=-1`，`hand_dof_pos[:,7:]=0`；身体 DOF 保持生成末帧 |

`0.7 s` 实际是 `max(1,round(0.7*F))/F`。例如 `F=25` 时 `H=R=18`，各为 `0.72 s`；下落首帧已过 `1/F` 秒，位置不是从零位移重复一帧开始。

### 3.3 后续追加的 `sparse_command` 与 `validation`

| 完整 report 路径 | 公式 / 含义 |
| --- | --- |
| `sparse_command.nonzero_translation_frames` | `sum(command[:,0]!=0)` |
| `sparse_command.nonzero_yaw_frames` | `sum(command[:,1]!=0)` |
| `sparse_command.combined_command_frames` | `sum((dx!=0)&(dyaw!=0))` |
| `sparse_command.release_frames` | `T-b=R` |
| `sparse_command.turn_stages` | 有效 turn stage 列表；每个包含下列三项 |
| `sparse_command.turn_stages[].start_frame` | 阶段起点，全局输出帧号 |
| `sparse_command.turn_stages[].end_frame_exclusive` | 阶段终点，不包含；yaw 端点用于算总角度 |
| `sparse_command.turn_stages[].total_yaw_rad` | `unwrap_yaw(end)-unwrap_yaw(start)`，未经过正负 `1.5π` 命令裁剪的原始总转角 |
| `validation.motion_key` | 输入检查函数收到的文件名主干 `name` |
| `validation.robot_key` | robot PKL 内部 motion key 的字符串 |
| `validation.object_key` | objects PKL 内部 motion key 的字符串 |
| `validation.total_frames` | 实际 robot DOF 数组的长度 |
| `validation.fps` | 实际 robot FPS；检查与物体 FPS 一致 |
| `validation.usd_path` | 找到的配对 USD/USDA 路径 |

`validation` 代表配对文件存在、基础数组形状/FPS/有限数值等输入检查通过；不会证明接缝平滑、抓握成功、无穿模、无碰撞，甚至不会验证所有 meta 边界彼此一致。当前源码在用户贴出的 report 字典之后还追加了 `grasp_orientation_policy`、`sparse_command`、`right_hand_opening_policy`、`validation`，不能漏掉。

本地旧 report 还可能有 `grail_end_frame_exclusive` 与旧字典 `sparse_commands`：后者的 `nonzero_translation_frames` / `nonzero_turn_frames` / `combined_command_frames` 是历史非零命令统计，当前实现使用单数 `sparse_command` 和 `nonzero_yaw_frames`；不应把旧字段当作现生成器再次计算出的结果。

## 4. 真正影响三个入口的读取链与计算

### 4.1 教师 residual 的渐入/渐出：训与师共用

源码：`EvalResidualTransition.weights()/apply()`。令当前绝对 motion 步数 `s=motion_start_time_steps+time_steps`、控制频率 `F_sim=motion_lib._sim_fps`、源帧等价位置 `f=s*F/F_sim`，则：

```text
w_in(f)  = clamp((f-transition_start_frame)
                 /(transition_end_frame-transition_start_frame),0,1)
w_out(f) = 1-clamp((f-residual_fade_out_start_frame)
                   /(residual_fade_out_end_frame-residual_fade_out_start_frame),0,1)
w(f)     = w_in(f)*w_out(f)
```

缺少完整渐入对时 `w_in=1`；渐出两个 key 均缺失时 `w_out=1`，出现任意一个时必须成对，且为整数满足 `0<=start<end<total_frames`。只对前 `tokenizer_action_dim` 维 residual 乘 `w`，手部动作通道不乘。另有配置的 residual scale，例如 pnp 当前为 `0.1`。

训练在 rollout 时保存权重，之后教师在 minibatch 上重新推理时使用当时权重生成蒸馏目标。学生输出完整 latent，不直接把学生完整 latent 乘 `w`。渐出后为零的是教师 GRAIL residual，不是关闭 SONIC 基础运动，也不是冻结机器人。

### 4.2 `place_carry_source_frame` 是训练 BC mask 的真实输入

`robocasa_carry_table` 继承的 `joint_micro_step` 开启 `teacher_termination_loss_override.enabled=True`。当前实现：

```text
carry_step = ceil(place_carry_source_frame * F_sim/F)
before_carry = (motion_start_time_steps+time_steps) < carry_step
masked = teacher_exceeded & ~student_exceeded & before_carry
valid = ~masked
BC = BC_coef * Σ(valid_i * per_sample_loss_i) / max(Σvalid_i,1)
```

这里的 exceeded 检查是教师/学生的 `ee_body_pos`、`object_pos_deviation`。carry 帧本身不属于 `before_carry`。没有 carry 字段时边界设为无穷大，保留旧屏蔽行为；字段存在时还会验证 `0<=c<total_frames`。因此不能把所有 `place_*source*` 都归类为纯记录。

### 4.3 `grail_start_frame` 与生成阶段边界的实际用途

motion library 会对 FPS 转换和片段裁剪对齐边界。设加载的第 `k` 帧对应源索引：

```text
I[k] = min(crop_start + floor(k*F/F_target + 1e-9), crop_end-1)
a_loaded = searchsorted(I, place_generated_start_frame)
g_loaded = searchsorted(I, grail_start_frame)
```

训练当前打开 `load_carry_target_pose`、`load_place_generated_phase`、`load_grail_tracking_phase`：

| 阶段 | 当前 carry-table 配置中的作用 |
| --- | --- |
| `step<a_loaded` | carry 位置成功奖励权重 `2.0`，成功条件为距离 `<0.1 m`；首参考接触之后 carry 距离代价权重 `-1.5`，接触条件比较使用严格 `>` |
| `step>=a_loaded` | release 目标阶段，成功奖励权重 `10.0`；`ee_body_pos` / `object_pos_deviation` 使用生成阶段阈值 `1.0`；关闭 approach velocity penalty |
| `g_loaded<=step<a_loaded` | 配置中 8 项上/下肢相对位置、姿态、线速度、角速度 tracking reward 置零；根部 anchor tracking 不是这组 gate |
| 生成开始且 `drop<0.5` | release 目标三维距离代价权重 `-0.5`，不要求参考/实际接触 |
| `drop>=0.5` | 关闭物体参考偏差终止和相应 grasp reward；release 成功还要求释放前最后一步曾进入目标球，随后确认实际物理释放 |

这些是配置的 reward weights；Isaac 奖励管理器另按控制周期汇总。`place_object_goal_quat_w` 影响姿态观测，上表的位置距离成功条件不含四元数误差。

运行时 `is_release_phase()` 的公式是 `get_sparse_policy_commands()[1]>=0.5`，**没有读取 `place_release_start_frame`**。只改 meta 的 release 帧而不重建 robot drop 数组，不会自动移动实际运行释放请求。

### 4.4 目标位姿如何成为学生输入

目标位置先跟随 motion library 的可选 `zero_root_xy` 平移，再加环境 origin 与 `object_z_offset`：

```text
p_goal_sim = p_goal_meta - [initial_root_x,initial_root_y,0] (仅 zero_root_xy=True)
             + env_origin + [0,0,object_z_offset]
p_goal_in_A = Q_A^T (p_goal_sim-p_A)
Q_goal_in_A = Q_A^T Q_goal
```

对当前仿真的 pelvis 和手部参考系分别计算一次，每个输出 `3D position + 6D rotation`，两个参考系共 18 维。6D 旋转取旋转矩阵前两列，并按当前实现 reshape。

carry-table 学生 privileged 输入是基础 48 + release goal 18 + carry goal 18 + sparse 2 + drop 1 = **87 维**。目标世界位姿从 meta 读取并缓存，相对位姿每步按实际仿真状态重算。`init_z_offset=object_init_z_offset=0.05` 只抬高 reset 写入仿真的初始位置；当前 `object_z_offset=0`，目标点不会因此也加 5 cm。

sparse 命令还经过真实接触状态门控：连续 3 步有手接触、物体不接触桌子且尚未请求释放后确认 pickup；只有 carrying 且 drop<0.5 才输出离线 x/yaw。默认力阈值 `0.1 N`。接触消失/释放/掉落由真实力维护，不能用 meta 接触统计替代。

### 4.5 用户给出的 reference-free 推理读哪些 meta

`eval_agent_trl.py:475` 检测到 `reference_free.enabled=True` 后调用独立 `run()` 并返回。因此既不创建原参考 motion library，也不运行 `EvalResidualTransition`。

此分支从 `dataset_dir/meta` 只直接读取：

```text
table_pos
table_quat
table_size
```

它另从 objects PKL 读取第一帧物体位置/姿态，从独立 `initial_pose_file` 读取机器人 `dof`、`hand_dof_pos`、`root_rot`、`root_trans_offset`，并加载 USD/BPS。这些不是 meta key。

若 checkpoint 的 privileged 输入为 87 维，则 carry/release 目标和 sparse/drop 来自 `sample_carry_plan()` 与 `OfflineCarryTask`。设重采样后的物体起始点为 `p0`、抬升为 `lift`、下放为 `lower`，规划根部轨迹为 `P`，起终 heading 为 `ψ0,ψ1`，则：

```text
p_carry = p0 + [0,0,lift]
p_release.xy = p_carry.xy + P_end
               + Rz(ψ1)*[0.33,-0.11] - Rz(ψ0)*[0.39,-0.09]
p_release.z = p_carry.z-lower
Q_release = Rz(ψ1-ψ0)*Q_object_initial
```

当前规划器默认 `lift∈[0.20,0.35]m`、`lower∈[0.05,0.14]m`、搬运距离 `[2.5,3.2]m`、命令 25 Hz。它不使用 meta 的 carry/release 目标、源帧边界、残差渐出或 `sparse_command_parameters`。若 checkpoint 是 66 维则走兼容的仅抬升目标分支，不启用 carry planner；87/66 的选择需以用户实际 checkpoint 旁 `config.yaml` 为准。

`episode_length_s=13` 只是仿真 episode 上限，不是将 meta 的 `total_frames/F` 重采样为 13 s；`terminate_on_success=False` 不改变上述读取范围。

## 5. Teacher rollout 对字段语义的改变

开启导出后代码强制 `run_once=True`、首帧初始化、关闭抓握前采样/随机起点并将 `eval_start_frame=0`。保存第一个 episode 的 action 执行后、自动 reset 前状态，包括终止状态；不按成功筛选。

### 5.1 采样与 meta 改写公式

设采集到 `L` 个控制步，控制周期 `dt`，输出 FPS 仍为源 `F`：

```text
T' = ceil(L*dt*F - 1e-9)
sim_index[k] = min(round((k+1)/(F*dt))-1, L-1), k=0..T'-1
source_index[k] = min(k, source_T-1)
```

robot 和 object 使用完全相同的 `sim_index`；位置减 env origin，robot 四元数转 xyzw、物体保持 wxyz。robot sparse/drop、SMPL 占位和 objects 局部接触标签按 `source_index` 继承；没有根据实际 teacher 根轨迹/实际接触重新生成 sparse 或接触标签。

| 字段 | teacher 保存行为 |
| --- | --- |
| `total_frames` | 改为 `T'` |
| 顶层 key 以 `_frame` 结尾且不含 `source` | `min(old_value,T'-1)`，包括渐入/渐出边界、`grail_start_frame`、place 阶段边界、output contact 首尾 |
| 顶层 key 以 `_frame_exclusive` 结尾 | `min(old_value,T')` |
| `place_object_goal_pos_w/quat_w` | `b'=min(old_release_start,T'-1)` 后，取实际导出 objects 的 `max(b'-1,0)` 帧 |
| `place_object_carry_goal_pos_w/quat_w` | 取实际导出 objects 的 `min(place_carry_source_frame,T'-1)` 帧 |
| 含 `source` 的来源帧、接触 duration/count/start_s、嵌套参数/turn stats | 原样保留；不会统一重算或递归裁剪 |

所以 rollout 数据中的 release goal 是**teacher 当时真正到达的位置**，不再必然等于原 Kimodo 约束目标。即便 teacher 失败，保存的 goal 也会改成失败轨迹对应帧的物体位姿。学习这些数据前，应清楚目标已被重标。

提前结束时的裁剪也不保证渐入/渐出边界仍严格递增；例如渐出 start/end 都变成 `T'-1`，后续 `EvalResidualTransition` 会拒绝。`place_carry_source_frame` 保留旧值而可能 `>=T'`，同样会被训练 BC boundary 检查拒绝。`place_turn_slowdown`、`sparse_command_parameters.goal_frame`、接触累计统计可能仍指向原时间轴。此处是现有导出行为，本文没有修改算法或数据。

### 5.2 teacher 对 report 的修改与新增 `rollout` 子项

teacher 只改 report 顶层 `total_frames/fps`、物体平移/旋转 policy，更新 `validation.total_frames/fps/usd_path`，再追加 `rollout`；原 `actual_root_distance_m`、`actual_hand_lower_m`、接触统计、sparse 统计、hold 长度等不重新计算。它们仍描述原拼接输入，不自动变成实际仿真测量值。

`RolloutReport` 则在**每个控制步**累计真实仿真量。以下 `p_k` 是真实根部位置，`θ_k` 是根部 yaw，初始状态记为 `k=0`，物体位置为 `o_k`，原始 release 目标为 `q*`。

| report 路径 | 物理意义与公式 |
| --- | --- |
| `rollout.turn_direction` | 对累计 yaw `Θ=ΣwrapToPi(θ_k-θ_(k-1))` 分类：`Θ>0.02` left、`Θ<-0.02` right，其余 straight |
| `rollout.reference_turn_direction` | 对 `meta["place_turn_direction"]` 用相同阈值分类；这就是额外 key 的直接依赖 |
| `rollout.yaw_change_rad` | `Θ`，实际有符号累计转角，不是首尾 quaternion 的最短角距离 |
| `rollout.release_goal_pos` | episode 开始缓存的**原输入 release 目标**，`get_release_target_pose()-env_origin`；不是导出时重标后的 meta 目标 |
| `rollout.min_release_goal_distance_m` | 包含初始状态在内的 `min_k ||o_k-q*||₂`，针对原目标，不表示在请求释放那一步成功 |
| `rollout.foot_ground_contacts.left` | 左脚落地事件数：接触从 false→true 计一次；首次观测到有接触也计一次 |
| `rollout.foot_ground_contacts.right` | 右脚同上 |
| `rollout.foot_ground_contacts.total` | 左右事件数之和，不是接触帧数或力积分 |
| `rollout.base_path_length_m` | `Σ||p_k-p_(k-1)||₂`，实际三维累计路径长度 |
| `rollout.base_xy_path_length_m` | `Σ||p_k.xy-p_(k-1).xy||₂`，实际 XY 累计路径长度 |
| `rollout.robot_table_collision` | 整段是否曾有机器人—桌子传感器合力范数累计值 `>1.0 N`；逐步逻辑 OR |
| `rollout.contact_force_threshold_n` | `1.0`，脚落地和机器人碰桌统计阈值；与 sparse 抓握状态默认 `0.1 N` 不同 |

每条 report 的 `rollout` 还会写入数据集根目录 `rollout_report.json` 的同名条目。离线 `select_teacher_rollouts.py` 确实会读取 report 的 rollout 指标及 meta 的 `grail_start_frame/place_output_contact_start_frame` 来筛选 300 条；这是**数据准备**用途，不是 `train_agent_trl.py` 训练时读取 report。

## 6. 一个本地真实样例

`pickup_table_place_concat/meta/pickup_table__alcohol_10__000.pkl` 及同名 report：

| 量 | 本地值 | 解释 |
| --- | --- | --- |
| `F/S/c/n` | `25/362/310/311` | 源保留到 310（含） |
| 旧 residual 渐入 | `112→141` | 旧 walk→pickup 接缝 |
| `g` | `132` | 名义 GRAIL pickup 起点，不等于渐入起点 |
| 新 residual 渐出 | `310→313` | 覆盖保留端点、2 个 seam 帧和首生成帧 |
| seam | `[311,313)` | 311、312 两帧 |
| generated | `[313,459)` | 146 帧 |
| hold | `[459,477)` | 18 帧，0.72 s |
| release | `[477,495)` | 18 帧，0.72 s |
| release goal 采样帧 | `476` | 最后一帧仍持物的静止帧 |
| 输入接触标注 | `251..352`、102 帧、4.08 s | 包含后来裁掉的源尾部 |
| 输出接触标注 | `251..476`、226 帧、9.04 s | 新增持物段也复制局部接触标签 |

在同名本地 teacher rollout 中，release meta 位置由原始约 `[-1.8534,1.7162,1.0472]` 改成实际约 `[-1.0798,1.6380,0.0231]`；`report.rollout.release_goal_pos` 仍保留原目标，`min_release_goal_distance_m≈1.0733`。这直接说明导出 meta 的 goal 与用于评估 teacher 是否接近原任务目标的 report goal 是两个不同量。

## 7. 源码索引

| 核对内容 | 文件与关键位置 |
| --- | --- |
| place 拼接时间轴、meta、report、物体自由落体 | [concat.py](../grail/place_data_generater/concat.py)，`append()` 约 356–534 行 |
| sparse 标签、参数、阶段转角、转身减速 | 同上 `generate_sparse_commands()` 约 74–162 行、`slow_first_turn()` 约 179 行 |
| 源接触统计、carry/lower 选帧、距离采样 | [constraints.py](../grail/place_data_generater/constraints.py)，28–50、74–84、130–145 行 |
| 旧渐入字段来源 | [concat_walk_motion.py](../grail/walk_data_tool/concat_walk_motion.py)，477–488 行 |
| 当前训练实验的观测/阶段/reward 配置 | [robocasa_carry_table.yaml](../imports/SONIC/gear_sonic/config/exp/manager/universal_token/distill/robocasa_carry_table.yaml) |
| residual 权重与 carry BC 边界读取 | [eval_residual_transition.py](../imports/SONIC/gear_sonic/trl/modules/eval_residual_transition.py)，`PlaceCarryBoundary`、`EvalResidualTransition` |
| 教师蒸馏目标应用 residual 权重 | [ppo_trainer.py](../imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py)，约 2106 行 |
| BC teacher/student exceeded mask | [ppo_trainer_aux_loss.py](../imports/SONIC/gear_sonic/trl/trainer/ppo_trainer_aux_loss.py)，约 164–178、311–326 行 |
| 目标/阶段读取与重采样 | [motion_lib_base.py](../imports/SONIC/gear_sonic/utils/motion_lib/motion_lib_base.py)，63–95、721–764、1844–1854、2460–2486 行 |
| table meta、目标 origin、释放判断 | [commands.py](../imports/SONIC/gear_sonic/envs/manager_env/mdp/commands.py)，2980–3021、3642–3681、3712–3793 行 |
| 18D 相对目标观测和 sparse/drop 观测 | [observations.py](../imports/SONIC/gear_sonic/envs/manager_env/mdp/observations.py)，3031–3092 行 |
| tracking gates、目标成功/距离奖励 | [rewards.py](../imports/SONIC/gear_sonic/envs/manager_env/mdp/rewards.py)，484、818–915 行 |
| 无参考推理提前路由、teacher 导出开关 | [eval_agent_trl.py](../imports/SONIC/gear_sonic/eval_agent_trl.py)，475–507 行 |
| 无参考输入实际读取的三个 meta key | [reference_free_data.py](../imports/SONIC/gear_sonic/utils/reference_free_data.py)，11–44 行 |
| 无参考 manager 替换及 checkpoint 维度分支 | [reference_free_eval.py](../imports/SONIC/gear_sonic/reference_free_eval.py)，`prepare_config()` |
| 无参考独立目标与 sparse 规划 | [reference_free_plan.py](../imports/SONIC/gear_sonic/utils/reference_free_plan.py)，`plan_carry_task()` / `sample_carry_plan()` |
| teacher 采样、meta/report 改写和实际指标 | [teacher_rollout.py](../imports/SONIC/gear_sonic/utils/teacher_rollout.py)，`RolloutReport`、`TeacherRolloutExporter.save()` |
| validation 的实际检查范围 | [batch_render_replay_clip.py](../grail/datatool/batch_render_replay_clip.py)，598–718 行 |
| 300 条 teacher 数据的离线筛选读取 | [select_teacher_rollouts.py](../grail/place_data_generater/select_teacher_rollouts.py)，`read_candidate()` |
