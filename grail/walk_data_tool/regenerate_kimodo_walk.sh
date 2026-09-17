#!/usr/bin/env bash
# 上传约束到 wostep → 按 generate_commands.sh 生成 → 下载 → 按 manifest 裁剪并拼接。
# 用法：bash grail/walk_data_tool/regenerate_kimodo_walk.sh [约束目录或 JSON] [--max-success=17] [--dry-run] [--render]
# 默认批量处理 kimodo_end_frame_last_step；--dry-run 不连接服务器、不修改数据。
# 默认到拼接结束，--render 追加渲染；保留 --skip-render 兼容旧调用。
# 批量模式单条失败后继续，最后汇总并返回非零退出码。
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
script="$repo/grail/walk_data_tool/regenerate_kimodo_walk.sh"
json="$repo/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/kimodo_end_frame_last_step"
grail_python=/home/tide/miniconda3/envs/grail/bin/python
dry_run=false
skip_render=true
json_given=false
max_success=0
for arg in "$@"; do
  case "$arg" in
    --max-success=*)
      max_success=${arg#*=}
      [[ "$max_success" =~ ^(0|[1-9][0-9]{0,5})$ ]] || {
        echo "--max-success 必须为 0 到 999999 的整数（0 表示全部）。" >&2
        exit 1
      } ;;
    --dry-run) dry_run=true ;;
    --skip-render) skip_render=true ;;
    --render) skip_render=false ;;
    -h|--help)
      echo "用法：bash $0 [约束目录或 JSON] [--max-success=17] [--dry-run] [--render]"
      echo "默认上传 kimodo_end_frame_last_step 的约束到服务器 wostep，生成、回传并拼接。"
      echo "--max-success=N：本次成功拼接 N 条后停止；默认 0，处理全部。"
      exit 0 ;;
    -*) echo "未知参数：$arg" >&2; exit 1 ;;
    *)
      if "$json_given"; then
        echo "一次只能指定一个约束目录或 JSON。" >&2
        exit 1
      fi
      # JSON 可以是链接，但必须使用所选目录旁的 manifest，不能切换到链接目标目录。
      json=$(realpath -e -s -- "$arg")
      json_given=true ;;
  esac
done
cd "$repo"
[[ -x "$grail_python" ]] || { echo "缺少 grail Python：$grail_python" >&2; exit 1; }

if [[ -d "$json" ]]; then
  # 使用成功记录枚举，避免把 manifest.json、failures.json 或旧约束当作动作。
  constraint_files=$(python3 - "$json" <<'PY'
import json
import re
import sys
from pathlib import Path

directory = Path(sys.argv[1]).resolve()
records = json.loads((directory / "manifest.json").read_text())["records"]
names = [record["constraint"] for record in records]
if not names or len(names) != len(set(names)):
    sys.exit("错误：manifest 必须包含非空且不重名的约束记录。")
for name in names:
    if not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*\.json", name):
        sys.exit(f"错误：非法约束文件名：{name}")
    if not (directory / name).is_file():
        sys.exit(f"错误：缺少约束文件：{directory / name}")
    print(directory / name)
PY
)
  mapfile -t files <<< "$constraint_files"
  options=()
  if "$dry_run"; then options+=(--dry-run); fi
  if ! "$skip_render"; then options+=(--render); fi
  failed=()
  succeeded=0
  attempted=0
  for file in "${files[@]}"; do
    attempted=$((attempted + 1))
    if bash "$script" "$file" "${options[@]}"; then
      succeeded=$((succeeded + 1))
      echo "批量进度：成功 $succeeded，失败 ${#failed[@]}，已处理 $attempted/${#files[@]}"
      if ((max_success > 0 && succeeded >= max_success)); then
        echo "已达到本次成功目标 $max_success，停止批量处理。"
        break
      fi
    else
      failed+=("$(basename "$file" .json)")
    fi
  done
  echo "批量完成：已处理 $attempted/${#files[@]}，成功 $succeeded，失败 ${#failed[@]}"
  if ((max_success > 0 && succeeded < max_success)); then
    echo "未达到本次成功目标 $max_success。" >&2
    exit 1
  fi
  if ((${#failed[@]})); then
    printf '失败动作：%s\n' "${failed[@]}"
    exit 1
  fi
  exit 0
fi

name=$(basename "$json" .json)
if [[ "$json" != *.json || ! "$name" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]]; then
  echo "请传入动作约束 JSON，文件名仅支持字母、数字、下划线、点和横线。" >&2
  exit 1
fi
dataset=$(dirname "$(dirname "$json")")
commands="$(dirname "$json")/generate_commands.sh"
manifest="$(dirname "$json")/manifest.json"
output="$repo/data/hf_dataset/data_update/data/pickup_table_walk_concat"
for file in "$json" "$commands" "$manifest" "$dataset/robot/$name.pkl" "$dataset/objects/$name.pkl"; do
  [[ -f "$file" ]] || { echo "缺少文件：$file" >&2; exit 1; }
done

# 已通过 SSH 确认：宿主机目录挂载到 /workspace/kimodo。
# 实际容器名是 yuguanchao_kimodo（不是 yuaguanchao_kimodo）。
server=cuixinru@202.120.37.249
ssh_command=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p 9989)
rsync_shell='ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p 9989'
container=yuguanchao_kimodo
remote_dir=/home/cuixinru/data0/kimodo/data/wostep
container_dir=/workspace/kimodo/data/wostep

# 上传前复用拼接器的校验，确认 manifest 指向同一份源动作，并且裁剪后仍够过渡。
# 按约束文件名精确匹配 generate_commands.sh，不依赖其中原有的服务器目录；
# 保留文本、模型、duration、seed，仅将输入和输出路径改写到 wostep。
plan=$("$grail_python" - "$commands" "$json" "$manifest" "$dataset" "$container_dir" <<'PY'
import json
import shlex
import sys
from pathlib import Path

from grail.walk_data_tool.concat_walk_motion import (
    constraint_source_frame, crop_paired_motion, load_single,
)

commands, constraint, manifest_path, dataset, destination = sys.argv[1:]
constraint, dataset = Path(constraint), Path(dataset)
name = constraint.stem
manifest = json.loads(Path(manifest_path).read_text())
records = [record for record in manifest["records"] if record["constraint"] == constraint.name]
if len(records) != 1:
    sys.exit(f"错误：manifest 中 {constraint.name} 必须恰好有一条记录。")
record = records[0]
robot_path = dataset / "robot" / f"{name}.pkl"
robot_key, robot = load_single(robot_path)
object_key, objects = load_single(dataset / "objects" / f"{name}.pkl")
if (robot_key != object_key or record["motion_key"] != str(robot_key)
        or Path(record["source"]).resolve() != robot_path.resolve()):
    sys.exit("错误：约束记录与机器人/物体的源路径或 motion key 不匹配。")
source_frame = constraint_source_frame(manifest_path, robot_path, robot_key, robot)
transition_frames = manifest.get("transition_frames", 10)
if type(transition_frames) is not int or transition_frames < 1:
    sys.exit("错误：manifest transition_frames 必须为正整数。")
crop_paired_motion(robot, objects, source_frame, transition_frames)
json.loads(constraint.read_text())
matches = []
for line in Path(commands).read_text().replace("\\\n", "").splitlines():
    args = shlex.split(line, comments=True)
    if args[:3] != ["python", "-m", "kimodo.scripts.generate"]:
        continue
    if any(args[i] == "--constraints" and Path(args[i + 1]).name == constraint.name
           for i in range(len(args) - 1)):
        matches.append(args)
if len(matches) != 1:
    sys.exit(f"错误：{commands} 中匹配 {constraint.name} 的命令有 {len(matches)} 条，必须恰好一条。")
args = matches[0]
for option in ("--constraints", "--output", "--num_samples", "--model"):
    if args.count(option) != 1 or args.index(option) + 1 == len(args):
        sys.exit(f"错误：目标命令必须包含唯一且有值的 {option}。")
if args[args.index("--num_samples") + 1] != "1":
    sys.exit("错误：单条更新要求 generate_commands.sh 中 --num_samples 为 1。")
if args[args.index("--model") + 1] != "Kimodo-G1-RP-v1":
    sys.exit("错误：该拼接流程要求 Kimodo-G1-RP-v1 的 G1 CSV 输出。")
args[args.index("--constraints") + 1] = f"{destination}/{name}.json"
args[args.index("--output") + 1] = f"{destination}/motions/{name}"
print(source_frame)
print(transition_frames)
print(shlex.join(args))
PY
)
mapfile -t plan_lines <<< "$plan"
source_frame=${plan_lines[0]}
transition_frames=${plan_lines[1]}
generate_command=${plan_lines[2]}
concat_command=(python grail/walk_data_tool/concat_walk_motion.py
  --walk-csv "$dataset/kimodo_walk/$name.csv"
  --dataset-dir "$dataset" --motion-name "$name" --output-dir "$output"
  --constraint-manifest "$manifest" --transition-frames "$transition_frames" --overwrite)

echo "动作：$name"
echo "命令来源：$commands"
echo "服务器工作目录：$remote_dir"
echo "$generate_command"
echo "下载覆盖：$dataset/kimodo_walk/$name.{csv,npz}"
echo "裁剪依据：$manifest；丢弃原动作 [0, $source_frame)，从第 $source_frame 帧接入"
echo "拼接输出：$output（仅 $name，transition-frames=$transition_frames）"
printf '拼接命令：'; printf '%q ' "${concat_command[@]}"; printf '\n'
if ! "$skip_render"; then echo "视频覆盖：$output/vis/$name.mp4"; fi
if "$dry_run"; then
  exit 0
fi
steps=4
if ! "$skip_render"; then steps=5; fi

# 和 concat_kimodo_walk.sh 一样使用本机 grail 环境。
set +u
source /home/tide/miniconda3/etc/profile.d/conda.sh
conda activate grail
set -u
for tool in ssh rsync python; do
  command -v "$tool" >/dev/null || { echo "缺少命令：$tool" >&2; exit 1; }
done

echo "[1/$steps] 上传约束到 $remote_dir"
"${ssh_command[@]}" "$server" /bin/bash -s <<REMOTE
set -euo pipefail
if [ "\$(docker inspect -f '{{.State.Running}}' '$container')" != true ]; then
  docker start '$container'
fi
# wostep 可能由容器 root 创建；仅调整本任务的两个目录，不递归修改其他数据。
docker exec '$container' mkdir -p '$container_dir/motions'
if [[ ! -w '$remote_dir' || ! -w '$remote_dir/motions' ]]; then
  docker exec '$container' chown "\$(id -u):\$(id -g)" '$container_dir' '$container_dir/motions'
fi
REMOTE
rsync -av --checksum -e "$rsync_shell" -- "$json" "$server:$remote_dir/$name.json"

echo "[2/$steps] 在 Docker 中运行 generate_commands.sh 的对应命令"
# 自动化使用 -i 接收脚本，不使用需要终端的 -t。
{
  cat <<'REMOTE'
set -eo pipefail
cd /workspace/kimodo
source /workspace/kimodo/deployment/env.sh
set -u
export HF_HUB_OFFLINE=1
export TEXT_ENCODER_MODE=local
export TEXT_ENCODER_DTYPE=float32
# 8B 文本编码器的 float32 权重超过共享 GPU 的剩余显存；动作模型仍使用 GPU。
export TEXT_ENCODER_DEVICE=cpu
REMOTE
  # 固定目录重复生成时，仅清理本动作的旧产物，防止误下载旧文件。
  printf "rm -f -- '%s/motions/%s.csv' '%s/motions/%s.npz'\n" \
    "$container_dir" "$name" "$container_dir" "$name"
  printf '%s\n' "$generate_command"
  printf "test -s '%s/motions/%s.csv'\n" "$container_dir" "$name"
  printf "test -s '%s/motions/%s.npz'\n" "$container_dir" "$name"
} | "${ssh_command[@]}" "$server" "docker exec -i '$container' /bin/bash -s"

# 先下载到临时目录，两个文件都成功取回后才覆盖本地文件。
work=$(mktemp -d /tmp/grail-kimodo-single.XXXXXXXX)
trap 'rm -rf -- "$work" "/tmp/vis_shard_$(basename "$work")"' EXIT
echo "[3/$steps] 从 wostep/motions 下载 CSV 和 NPZ，覆盖同名旧版本"
rsync -av --checksum -e "$rsync_shell" -- \
  "$server:$remote_dir/motions/$name.csv" \
  "$server:$remote_dir/motions/$name.npz" "$work/"
test -s "$work/$name.csv"
test -s "$work/$name.npz"
mkdir -p "$dataset/kimodo_walk"
mv -f -- "$work/$name.csv" "$work/$name.npz" "$dataset/kimodo_walk/"

echo "[4/$steps] 裁剪原动作并拼接到 $output"
if [[ -d "$dataset/object_usd/textures" ]]; then
  mkdir -p "$output/object_usd"
  cp -a "$dataset/object_usd/textures" "$output/object_usd/"
fi
"${concat_command[@]}"
if "$skip_render"; then
  echo "完成：$output/robot/$name.pkl（已跳过渲染）"
  exit 0
fi

echo "[5/5] 仅重新渲染当前动作"
# visualize.sh 会抽样且 --skip_existing；用新的单动作视图确保重渲染。
# robot 只链接目标文件，其余目录整体链接，以保留 USD 相对资源路径。
mkdir -p "$work/robot"
ln -s "$output/robot/$name.pkl" "$work/robot/$name.pkl"
for dir in objects object_usd meta; do
  ln -s "$output/$dir" "$work/$dir"
done
# 0 表示渲染视图中的全部（这里只含一条），同时关闭合集/网格后处理。
CUDA_VISIBLE_DEVICES=0 bash grail/visualization/scripts/visualize.sh \
  "$work" 0 1.5,-1.5,1.0 xyzw local 1
test -s "$work/vis/$name.mp4"
mkdir -p "$output/vis"
mv -f -- "$work/vis/$name.mp4" "$output/vis/$name.mp4"
echo "完成：$output/vis/$name.mp4"
