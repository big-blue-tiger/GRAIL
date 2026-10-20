#!/usr/bin/env bash
set -euo pipefail

# Save this file in /home/GRAIL and run: bash run_sparse_command_vis.sh
cd "$(dirname "${BASH_SOURCE[0]}")"

DATASET="${PWD}/data/hf_dataset/data_update/data/pickup_table_teacher_rollout"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"

for tool in ffmpeg ffprobe; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "Missing dependency: $tool (install with: apt-get update && apt-get install -y ffmpeg)" >&2
        exit 1
    }
done

if [[ -x /workspace/isaaclab/_isaac_sim/python.sh ]]; then
    PYTHON_CMD=(/workspace/isaaclab/_isaac_sim/python.sh)
    RUNTIME=server
else
    PYTHON_CMD=(python)
    RUNTIME=local
fi

# Forward extra CLI options, e.g. --motion MOTION_STEM or --existing-only.
"${PYTHON_CMD[@]}" -u grail/visualization/batch_sparse_command_vis.py \
    "$DATASET" \
    --output-dir "$DATASET/vis_sparse_command" \
    --max-videos "$MAX_VIDEOS" \
    --skip-existing \
    --camera-offset="1.5,-1.5,1.0" \
    --quat-convention xyzw \
    --runtime "$RUNTIME" \
    --parallel-jobs "$PARALLEL_JOBS" \
    "$@"
