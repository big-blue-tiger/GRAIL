#!/usr/bin/env bash
# 在 yuguanchao_sonic 中运行，仅拼接 manifest 中的全部动作，不启动仿真或渲染。
set -euo pipefail

cd /home/GRAIL
dataset=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat
output=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat
shared=/home/kimodo/data/wostep/place_walk

/isaac-sim/python.sh - "$dataset" "$output" "$shared" <<'PY'
import json
from pathlib import Path
import sys
from time import perf_counter

from grail.place_data_generater.concat import append
from grail.walk_data_tool.export_kimodo_end_frame import load_kinematics

dataset, output, shared = sys.argv[1:]
manifest = Path(shared) / "manifest.json"
with manifest.open() as file:
    records = json.load(file)["records"]
if not records:
    raise SystemExit("No constraints in manifest")

print(f"Concat only (no rendering): {len(records)} motions; load kinematics once", flush=True)
kinematics = load_kinematics(Path("/home/kimodo"))
batch_start = perf_counter()
for index, record in enumerate(records, 1):
    name = record["name"]
    csv = Path(shared) / "motions" / f"{name}.csv"
    start = perf_counter()
    print(f"Concat [{index}/{len(records)}]: {name}", flush=True)
    report = append(
        dataset, output, name, manifest, csv,
        kimodo_repo=Path("/home/kimodo"), overwrite=True,
        kinematics=kinematics, copy_shared_assets=index == 1,
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"Done [{index}/{len(records)}]: {name} ({perf_counter() - start:.2f}s)", flush=True)
print(f"Concat complete: {len(records)} motions ({perf_counter() - batch_start:.2f}s)", flush=True)
PY
