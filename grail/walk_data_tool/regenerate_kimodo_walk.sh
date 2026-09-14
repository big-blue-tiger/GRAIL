#!/usr/bin/env bash
# 单条动作：上传约束 → 服务器生成 → 下载覆盖 → 拼接 → 重新渲染。
# 用法：bash grail/walk_data_tool/regenerate_kimodo_walk.sh [约束 JSON] [--dry-run]
# 不传 JSON 时处理 pickup_table__alcohol_1__001；--dry-run 只检查并打印命令。
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
json="$repo/data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/kimodo_end_frame/pickup_table__alcohol_1__001.json"
dry_run=false
json_given=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    -h|--help)
      echo "用法：bash $0 [约束 JSON] [--dry-run]"
      exit 0 ;;
    -*) echo "未知参数：$arg" >&2; exit 1 ;;
    *)
      if "$json_given"; then
        echo "一次只能处理一个 JSON。" >&2
        exit 1
      fi
      json=$(realpath -e -- "$arg")
      json_given=true ;;
  esac
done
cd "$repo"

name=$(basename "$json" .json)
if [[ "$json" != *.json || ! "$name" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]]; then
  echo "请传入动作约束 JSON，文件名仅支持字母、数字、下划线、点和横线。" >&2
  exit 1
fi
dataset=$(dirname "$(dirname "$json")")
commands="$(dirname "$json")/generate_commands.sh"
output="$repo/data/hf_dataset/data_update/data/pickup_table_walk_concat"
for file in "$json" "$commands" "$dataset/robot/$name.pkl"; do
  [[ -f "$file" ]] || { echo "缺少文件：$file" >&2; exit 1; }
done

# 已通过 SSH 确认：宿主机目录挂载到 /workspace/kimodo。
# 实际容器名是 yuguanchao_kimodo（不是 yuaguanchao_kimodo）。
server=cuixinru@202.120.37.249
container=yuguanchao_kimodo
# 每次生成使用新目录，避免生成失败后误下载上次残留的结果。
run="grail_single/$name/$(date +%Y%m%d_%H%M%S)_$$"
remote_dir="/home/cuixinru/data0/kimodo/data/$run"
container_dir="/workspace/kimodo/data/$run"

# shlex 只解析命令文本，不执行整个 generate_commands.sh。
# 精确匹配 --constraints 的完整路径，仅改写输入和输出路径；
# 文本、模型、duration、seed 等参数均取自匹配到的那条命令。
generate_command=$(python3 - "$commands" "$name" "$container_dir" <<'PY'
import shlex
import sys
from pathlib import Path

commands, name, destination = sys.argv[1:]
expected = f"/workspace/kimodo/data/local_pickup/{name}.json"
matches = []
for line in Path(commands).read_text().replace("\\\n", "").splitlines():
    args = shlex.split(line, comments=True)
    if args[:3] != ["python", "-m", "kimodo.scripts.generate"]:
        continue
    if any(args[i:i + 2] == ["--constraints", expected] for i in range(len(args))):
        matches.append(args)
if len(matches) != 1:
    sys.exit(f"错误：{commands} 中匹配 {expected} 的命令有 {len(matches)} 条，必须恰好一条。")
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
print(shlex.join(args))
PY
)

echo "动作：$name"
echo "命令来源：$commands"
echo "服务器工作目录：$remote_dir"
echo "$generate_command"
echo "下载覆盖：$dataset/kimodo_walk/$name.{csv,npz}"
echo "拼接输出：$output（仅 $name，transition-frames=10）"
echo "视频覆盖：$output/vis/$name.mp4"
if "$dry_run"; then
  exit 0
fi

# 和 concat_kimodo_walk.sh 一样使用本机 grail 环境。
set +u
source /home/tide/miniconda3/etc/profile.d/conda.sh
conda activate grail
set -u
for tool in ssh rsync python; do
  command -v "$tool" >/dev/null || { echo "缺少命令：$tool" >&2; exit 1; }
done

echo "[1/5] 上传本次 JSON"
ssh -p 9989 "$server" "mkdir -p '$remote_dir/motions'"
rsync -av --checksum -e 'ssh -p 9989' -- "$json" "$server:$remote_dir/$name.json"

echo "[2/5] 在 Docker 中运行匹配到的生成命令"
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
REMOTE
  printf '%s\n' "$generate_command"
  printf "test -s '%s/motions/%s.csv'\n" "$container_dir" "$name"
  printf "test -s '%s/motions/%s.npz'\n" "$container_dir" "$name"
} | ssh -p 9989 "$server" "docker exec -i '$container' /bin/bash -s"

# 先下载到临时目录，两个文件都成功取回后才覆盖本地文件。
work=$(mktemp -d /tmp/grail-kimodo-single.XXXXXXXX)
trap 'rm -rf -- "$work" "/tmp/vis_shard_$(basename "$work")"' EXIT
echo "[3/5] 下载 CSV 和 NPZ，覆盖同名旧版本"
rsync -av --checksum -e 'ssh -p 9989' -- \
  "$server:$remote_dir/motions/$name.csv" \
  "$server:$remote_dir/motions/$name.npz" "$work/"
test -s "$work/$name.csv"
test -s "$work/$name.npz"
mkdir -p "$dataset/kimodo_walk"
mv -f -- "$work/$name.csv" "$work/$name.npz" "$dataset/kimodo_walk/"

echo "[4/5] 仅重新拼接当前动作"
python grail/walk_data_tool/concat_walk_motion.py \
  --walk-csv "$dataset/kimodo_walk/$name.csv" \
  --dataset-dir "$dataset" \
  --motion-name "$name" \
  --output-dir "$output" \
  --transition-frames 10 \
  --overwrite

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
