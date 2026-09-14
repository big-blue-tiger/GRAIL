#!/usr/bin/env bash
set -eo pipefail
source /home/tide/miniconda3/etc/profile.d/conda.sh
conda activate grail
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
dataset=data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded
output=data/hf_dataset/data_update/data/pickup_table_walk_concat

shopt -s failglob
files=("$dataset"/kimodo_walk/*.csv)
index=0
for csv in "${files[@]}"; do
  index=$((index + 1))
  name=$(basename "$csv" .csv)
  echo "[$index/${#files[@]}] $name"
  python grail/walk_data_tool/concat_walk_motion.py \
    --walk-csv "$csv" \
    --dataset-dir "$dataset" \
    --motion-name "$name" \
    --output-dir "$output" \
    --transition-frames 10 \
    --overwrite
done
echo "完成：${#files[@]} 个动作已拼接到 $output"
