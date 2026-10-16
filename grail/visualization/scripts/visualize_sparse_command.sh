#!/usr/bin/env bash
# Reuse replay videos and render missing ones, then append sparse-command values.
# Usage: bash grail/visualization/scripts/visualize_sparse_command.sh DATASET [options]
# Options are documented by --help; requires a Python environment with joblib/numpy.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo"
python -m grail.visualization.batch_sparse_command_vis "$@"
