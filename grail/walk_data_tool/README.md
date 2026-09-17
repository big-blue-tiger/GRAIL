# 行走动作拼接

`concat_walk_motion.py` 将 Kimodo 行走动作接在原始抓取动作之前，并导出配套数据。下面统一使用一种批量写法：在仓库根目录的 Bash 终端执行，使用已安装项目依赖的 `grail` Conda 环境。

输入数据集为 `data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded`，遍历其中 `robot/*.pkl`，以文件名（不含扩展名）作为 `--motion-name`，查找 `kimodo_walk/` 中的同名 CSV。例如，`robot/pickup_table__alcohol_12__001.pkl` 对应 `kimodo_walk/pickup_table__alcohol_12__001.csv` 和 `objects/pickup_table__alcohol_12__001.pkl`，以及同名的配套资源。拼接读取 CSV，不读取旁边的 NPZ 文件。

## 按最后一次脚步结束帧生成约束

`export_kimodo_end_frame.py` 扫描**整段**原始 robot motion，利用 Kimodo 的 CPU
运动学计算左右脚踝、脚尖的世界坐标，选择最后一次有效移动后双脚稳定窗口的起始帧 `k`。
`k` 是原数据的零基帧索引，它作为 **Kimodo 生成行走的末帧约束**；生成行走的第一帧仍为
采样得到的起点。检测包含抓取期间、抓取之后的小碎步，因此允许裁掉部分抓取过程。

先导出到独立目录（不会加载生成模型、CUDA 或 IsaacSim）：

```bash
dataset=data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded
python grail/walk_data_tool/export_kimodo_end_frame.py \
  --input-dir "$dataset/robot" \
  --output-dir "$dataset/kimodo_end_frame_last_step" \
  --workers 4 --replay
```

默认最多使用 4 个 CPU 进程，按 PKL 分配任务，每个进程复用运动学实例并限制计算线程数。
`--workers 1` 可串行运行；同一 seed 下，改变进程数或 `--pattern` 不改变单条动作的结果。
重复写入已有输出需加 `--overwrite`。每个 PKL 可包含多个 motion，输出名会附加
`__motion_000` 等序号；后续现有拼接接口仍要求每个 PKL 只有一个 motion。

检测参数如下，时间均按**原动作 FPS**换算，`--fps` 仍只控制 Kimodo 生成 FPS：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--foot-speed-on` | `0.05` | 任一脚踝/脚尖开始移动的速度，m/s |
| `--foot-speed-off` | `0.03` | 双脚稳定窗口内的速度上限，m/s |
| `--min-foot-excursion` | `0.03` | 区间内相对起点的最大三维位移，m；不是净位移 |
| `--stable-time` | `0.3` | 确认稳定所需的连续时长，s；不额外后移选帧 |
| `--smooth-window` | `0.08` | 检测用中值滤波窗口，s；转换为奇数帧，0 禁用 |
| `--transition-frames` | `10` | 为后续 PCHIP 保留至少 N+1 帧，需与拼接参数一致 |

速度使用相邻帧差分，左右脚独立查找区间；稳定窗口同时检查双脚。有效位移包含原地抬脚后
回落、仅脚尖移动。小于位移阈值的区间作为噪声忽略；始终低于开始速度的缓慢移动不会触发
脚步区间。这是运动学检测，不是接触力判定，可结合诊断图调整阈值。滤波仅影响检测，导出
约束始终使用第 `k` 帧原始姿态。没有有效脚步且存在稳定状态时选第 0 帧。

默认参数已根据六条视频的人工停步时间校准。旧的 5 mm 门槛会将站立时脚跟抬落和踝部
晃动误判为小碎步；现在以 3 cm 为有效位移门槛，同时将启动/稳定速度调整为 0.05/0.03 m/s。
稳定观察时间为 0.3 s，以免把迈步中的短暂停顿当作结束；中值滤波仍为 0.08 s。
参数统一应用于所有动作，不按文件名指定帧。小于 3 cm 的微动默认忽略，仍可用 CLI 降低门槛。

以下时间均使用原始 25 FPS，视频帧数和 FPS 已核对一致；人工标注按近似秒数解释。
`alcohol_4__000` 已确认是第 4 秒结束、第 5 秒刚开始，目标记为 5.0 s。

| 动作（省略 `pickup_table__`） | 人工停步时间（s） | 新检测时间（s） | 原始帧 k |
| --- | --- | --- | --- |
| `alcohol_1__001` | 3.0 | 3.00 | 75 |
| `alcohol_2__000` | 1.0 | 1.44 | 36 |
| `alcohol_2__001` | 5.0 | 5.24 | 131 |
| `alcohol_2__004` | 0.0 | 0.00 | 0 |
| `alcohol_4__000` | 5.0 | 5.36 | 134 |
| `alcohol_4__001` | 2.0 | 1.92 | 48 |

校准样本最大偏差为 0.44 s，回归测试容差为 0.5 s，无走动样本要求严格选第 0 帧。
这些是校准集结果，不代表其他数据集上的检测准确率。

输出包括每条成功动作的约束 JSON、`manifest.json`、仅包含成功动作的 `generate_commands.sh`、
`failures.json`，以及 `diagnostics/<动作名>.png`。PNG 显示原始/滤波轨迹、速度、移动区间和
选帧；速度轴在阈值附近为线性、较大值为对数，便于同时检查大步和小碎步。`--replay` 额外
生成可直接用浏览器打开的 HTML，播放 `k` 前后各 1 秒的原始骨架，支持逐帧拖动、脚部放大；
无法选帧时显示末尾 1 秒。HTML 无网络、视频编码器或模拟器依赖。

末尾仍在移动、无法确认双脚稳定、剩余帧不足或单条数据异常时，记录失败并继续其余动作，
最后返回非零退出码。损坏的 PKL 也会生成说明失败原因的占位图。manifest 中的成功记录包含
`source_frame`、源文件 SHA-256、帧数、FPS、检测参数和移动区间。覆盖重跑时，本次失败动作
以前生成的约束 JSON 会移除，防止误用；其他未参与本次处理的文件不会清理。

**必须使用这些新约束重新运行 Kimodo 生成，不能把现有旧 CSV 直接用于新裁剪记录。**
将约束上传至 `--server-data-dir` 对应目录，在服务器运行新的 `generate_commands.sh`，再把
生成 CSV 按动作名放入 `kimodo_walk/`。模型生成沿用现有服务器流程，不在本地检测阶段执行。

## 使用同一 manifest 同步裁剪和拼接

批量脚本仍串行拼接，显式传入本次生成时使用的 manifest：

```bash
bash grail/walk_data_tool/concat_kimodo_walk.sh \
  data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded \
  data/hf_dataset/data_update/data/pickup_table_walk_concat \
  data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/kimodo_end_frame_last_step/manifest.json
```

拼接通过源路径和 motion key 查找唯一记录，并检查 SHA-256、帧数和 FPS。记录缺失、重复或
源文件变化均报错。机器人所有时序字段和物体位置/旋转从 `[k:]` 同步裁剪；删除 `< k` 的
接触记录，其余接触索引先减 `k`，再增加行走前缀长度。物体在新增前缀期间保持原第 `k` 帧
状态。直接调用 Python 接口或 CLI 时若省略 `--constraint-manifest`，仍按旧行为从第 0 帧拼接。

以下是等价的展开批处理命令：

```bash
(
  set -eo pipefail
  set -u
  shopt -s failglob

  dataset=data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded
  output=data/hf_dataset/data_update/data/pickup_table_walk_concat
  constraint_manifest="$dataset/kimodo_end_frame_last_step/manifest.json"

  mkdir -p "$output/object_usd/textures"
  cp -a "$dataset/object_usd/textures/." "$output/object_usd/textures/"

  mkdir -p "$output/reports"
  missing_list="$output/reports/missing_walk_csv.txt"
  failed_list="$output/reports/failed_concat.txt"
  : > "$missing_list"
  : > "$failed_list"
  total=0
  succeeded=0
  missing=0
  failed=0

  for robot in "$dataset"/robot/*.pkl; do
    name=$(basename "$robot" .pkl)
    csv="$dataset/kimodo_walk/$name.csv"
    total=$((total + 1))
    if [[ ! -f "$csv" ]]; then
      printf '%s\n' "$name" >> "$missing_list"
      missing=$((missing + 1))
      continue
    fi
    if python grail/walk_data_tool/concat_walk_motion.py \
      --dataset-dir "$dataset" \
      --walk-csv "$csv" \
      --motion-name "$name" \
      --output-dir "$output" \
      --constraint-manifest "$constraint_manifest" \
      --transition-frames 10 \
      --overwrite; then
      succeeded=$((succeeded + 1))
    else
      printf '%s\n' "$name" >> "$failed_list"
      failed=$((failed + 1))
    fi
  done

  printf '\n总数：%d，成功：%d，缺少 CSV：%d，转换失败：%d\n' \
    "$total" "$succeeded" "$missing" "$failed"
  printf '\n缺少 walk-csv 的动作（%s）：\n' "$missing_list"
  cat "$missing_list"
  printf '\n转换失败的动作（%s）：\n' "$failed_list"
  cat "$failed_list"
  # 有未成功转换的动作时，汇总后返回非零退出码。
  [[ "$missing" -eq 0 && "$failed" -eq 0 ]]
)
```

`--transition-frames 10` 使用四点 PCHIP 插值，新增 10 帧，并改写行走末尾和裁剪后动作开头各 10 帧的过渡区间；两段动作都至少需要 11 帧。行走段通过水平平移和偏航旋转对齐第 `k` 帧，按原始动作的 FPS 使用，不进行重采样。裁剪后第 10 帧（即原动作 `k+10` 帧）起保持原始数据不变。

`--overwrite` 允许批量处理时重复写入共享 BPS 资源，也会覆盖输出目录中已有的同名结果。源数据集保持不变；缺少 CSV 或单个动作转换失败时，记录动作名并继续处理其余动作，具体转换报错保留在终端输出中。最后打印本次总数、成功数、缺少 CSV 数和转换失败数，并列出未成功的动作名；清单分别保存到 `reports/missing_walk_csv.txt` 和 `reports/failed_concat.txt`，每次运行重置。只导出有对应 CSV 且转换成功的动作；输出目录中以前运行的结果不会自动清理，统计以本次执行为准。有未成功转换的动作时，命令在输出汇总后返回非零退出码。

全部结果写入 `data/hf_dataset/data_update/data/pickup_table_walk_concat`，每个动作包含以下完整配套文件：

- `robot/<动作名>.pkl`：拼接后的机器人动作。
- `objects/<动作名>.pkl`：与机器人帧数一致的物体动作及接触信息。
- `meta/<动作名>.pkl`：保留原始元数据，并更新总帧数、过渡起止帧（从 0 开始，包含两端）。
- `bps/<动作名>.npy` 和 `bps/_*.npy`：对应动作及共享 BPS 资源。
- `object_usd/<动作名>.usd` 或 `.usda`：原始物体资产。
- `object_usd/textures/`：由命令中的 `cp -a` 完整复制共享纹理目录，保留子目录结构和隐藏文件；重复执行时覆盖同名纹理文件。
- `reports/<动作名>.json`：输入来源、帧数、对齐误差及数据校验结果。

报告额外记录 `source_frame`、`source_num_frames` 和 `source_to_output_frame_offset`。
原始保留帧 `t` 对应输出帧 `t + source_to_output_frame_offset`；机器人过渡改写范围按原有
`replacement_start_index` / `replacement_stop_index_exclusive` 表示。

CPU 测试（包含本地 Kimodo 存在时的串并行一致性与真实 FK 拼接检查）：

```bash
python -m unittest discover -s grail/walk_data_tool/tests -v
```

每个动作必须具备上述输入配套资源，其中同名 USD/USDA 资产应恰好有一个。脚本在写出前会重新读取并校验动作及资源。

Kimodo 行走段的 `hand_action_left` / `hand_action_right` 在拼接时直接设为 `+1`（闭合），
之后沿用现有 PCHIP 过渡至原动作的手指令，无需额外后处理。`hand_dof_pos` 仍沿用原动作接入帧的记录。
