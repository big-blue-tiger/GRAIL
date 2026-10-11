#!/usr/bin/env bash
# Re-concat cached place motions using the local inputs from run.sh.
# Usage: bash grail/place_data_generater/run_concat_all.sh [--dry-run]
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
dry_run=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

set +u
source /home/tide/miniconda3/etc/profile.d/conda.sh
conda activate grail
set -u

python - "$repo" "$dry_run" <<'PY'
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from grail.place_data_generater.concat import append
from grail.place_data_generater.g1_kinematics import KINEMATICS_VERSION
from grail.place_data_generater.constraints import KIMODO
from grail.walk_data_tool.concat_walk_motion import file_hash
from grail.walk_data_tool.export_kimodo_end_frame import load_kinematics

repo = Path(sys.argv[1])
dataset = repo / "data/hf_dataset/data_update/data/pickup_table_walk_concat"
output = repo / "data/hf_dataset/data_update/data/pickup_table_place_left_right"
downloads = dataset / "kimodo_place"
names = sorted(path.stem for path in downloads.glob("*.csv"))
if not names:
    raise SystemExit(f"No cached place CSVs in {downloads}")

# Check every cached CSV and its source snapshot before overwriting any motion.
inputs = []
for name in names:
    manifest = dataset / "kimodo_place_constraints" / name / "manifest.json"
    csv = dataset / "kimodo_place" / f"{name}.csv"
    records = [r for r in json.loads(manifest.read_text())["records"] if r["name"] == name]
    if len(records) != 1:
        raise ValueError(f"{name}: expected exactly one manifest record")
    record = records[0]
    if record.get("kinematics_version") != KINEMATICS_VERSION:
        raise ValueError(f"{name}: cached constraints use incompatible kinematics")
    for directory, field in (("robot", "source_robot_sha256"),
                             ("objects", "source_objects_sha256")):
        if file_hash(dataset / directory / f"{name}.pkl") != record[field]:
            raise ValueError(f"{name}: source {directory} changed since generation")
    qpos = np.loadtxt(csv, delimiter=",", ndmin=2)
    if qpos.shape != (record["num_frames"], 36) or not np.isfinite(qpos).all():
        raise ValueError(f"{name}: invalid cached CSV shape/values: {qpos.shape}")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-3):
        raise ValueError(f"{name}: invalid cached CSV root quaternions")
    inputs.append((name, manifest, csv))
print(f"Preflight passed: {len(inputs)} cached place motions", flush=True)
if sys.argv[2] == "true":
    raise SystemExit(0)

kinematics = load_kinematics(KIMODO)
start = perf_counter()
for index, (name, manifest, csv) in enumerate(inputs, 1):
    report = append(dataset, output, name, manifest, csv, overwrite=True,
                    kinematics=kinematics, copy_shared_assets=index == 1)
    print(f"Concat [{index}/{len(inputs)}]: {name}; "
          f"{report['validation']['total_frames']} frames validated", flush=True)
print(f"Concat complete: {len(inputs)}/{len(inputs)} motions "
      f"({perf_counter() - start:.2f}s)", flush=True)
PY
