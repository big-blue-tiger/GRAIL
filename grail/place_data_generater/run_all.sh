#!/usr/bin/env bash
# Run the simple single-motion pipeline for every paired source motion.
# run.sh writes outputs to pickup_table_place_left_right.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dataset="$repo/data/hf_dataset/data_update/data/pickup_table_walk_concat"
failed=()
shopt -s nullglob
files=("$dataset"/robot/*.pkl)
((${#files[@]})) || { echo "No robot PKL files in $dataset" >&2; exit 1; }
for file in "${files[@]}"; do
  name=${file##*/}
  name=${name%.pkl}
  if ! bash "$repo/grail/place_data_generater/run.sh" "$name" "$@"; then
    failed+=("$name")
  fi
done
echo "Processed ${#files[@]} motions; failed ${#failed[@]}."
if ((${#failed[@]})); then
  printf 'Failed: %s\n' "${failed[@]}"
  exit 1
fi
