#!/usr/bin/env bash
set -eo pipefail
source /home/tide/miniconda3/etc/profile.d/conda.sh
conda activate grail
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
dataset=${1:-data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded}
output=${2:-data/hf_dataset/data_update/data/pickup_table_walk_concat}
manifest=${3:-$dataset/kimodo_end_frame/manifest.json}
[[ -f "$manifest" ]] || { echo "缺少约束 manifest: $manifest" >&2; exit 1; }
mkdir -p "$output/reports" "$output/object_usd/textures"
if [[ -d "$dataset/object_usd/textures" ]]; then
  cp -a "$dataset/object_usd/textures/." "$output/object_usd/textures/"
fi
failed_list="$output/reports/failed_concat.txt"
missing_list="$output/reports/missing_walk_csv.txt"
: > "$failed_list"
: > "$missing_list"

shopt -s failglob
files=("$dataset"/robot/*.pkl)
index=0
succeeded=0
failed=0
missing=0
for robot in "${files[@]}"; do
  index=$((index + 1))
  name=$(basename "$robot" .pkl)
  csv="$dataset/kimodo_walk/$name.csv"
  echo "[$index/${#files[@]}] $name"
  if [[ ! -f "$csv" ]]; then
    printf '%s\n' "$name" >> "$missing_list"
    missing=$((missing + 1))
    continue
  fi
  if python grail/walk_data_tool/concat_walk_motion.py \
    --walk-csv "$csv" \
    --dataset-dir "$dataset" \
    --motion-name "$name" \
    --output-dir "$output" \
    --constraint-manifest "$manifest" \
    --transition-frames 10 \
    --overwrite; then
    succeeded=$((succeeded + 1))
  else
    printf '%s\n' "$name" >> "$failed_list"
    failed=$((failed + 1))
  fi
done
echo "完成：总数 ${#files[@]}，成功 $succeeded，缺少 CSV $missing，失败 $failed；输出 $output"
[[ "$failed" -eq 0 && "$missing" -eq 0 ]]
