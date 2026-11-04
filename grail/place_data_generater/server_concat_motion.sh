#!/usr/bin/env bash
# 在 yuguanchao_sonic 中运行，仅拼接 manifest 中的全部动作，不启动仿真或渲染。
set -euo pipefail

cd /home/GRAIL
dataset=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat
output=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_place_concat_slow_turn
shared=/home/kimodo/data/wostep/place_walk_1018
turn_slowdown_factor=${TURN_SLOWDOWN_FACTOR:-1.2} # 首个转身段的帧数倍率;1 保持原速。

/isaac-sim/python.sh - "$dataset" "$output" "$shared" "$turn_slowdown_factor" <<'PY'
import json
from pathlib import Path
import sys
from time import perf_counter

from grail.place_data_generater.concat import append
from grail.walk_data_tool.export_kimodo_end_frame import load_kinematics

dataset, output, shared, turn_slowdown_factor = sys.argv[1:]
turn_slowdown_factor = float(turn_slowdown_factor)
manifest = Path(shared) / "manifest.json"
with manifest.open() as file:
    records = json.load(file)["records"]
if not records:
    raise SystemExit("No constraints in manifest")

print(f"Concat only (no rendering): {len(records)} motions; load kinematics once", flush=True)
kinematics = load_kinematics(Path("/home/kimodo"))
batch_start = perf_counter()
succeeded = 0
skipped = []
for index, record in enumerate(records, 1):
    name = record["name"]
    csv = Path(shared) / "motions" / f"{name}.csv"
    start = perf_counter()
    print(f"Concat [{index}/{len(records)}]: {name}", flush=True)
    try:
        motion_path = append(
            dataset, output, name, manifest, csv,
            kimodo_repo=Path("/home/kimodo"), overwrite=True,
            kinematics=kinematics, copy_shared_assets=succeeded == 0,
            turn_slowdown_factor=turn_slowdown_factor,
        )
    except ValueError as exc:
        # Skip invalid CSVs or missing/invalid carry-frame contacts; other failures stop the batch.
        csv_error = (f"{name}: expected finite ({record['num_frames']},36) Kimodo CSV "
                     "matching the manifest; got shape ")
        contact_error = (f"Carry frame {record['carry_source_frame']} must have "
                         "nonempty finite right-hand contact points")
        if not (str(exc).startswith(csv_error) or str(exc) == contact_error):
            raise
        # Delete this motion's output files, including leftovers from earlier runs.
        # Keep source PKLs, generated CSVs, manifest and shared assets intact.
        if Path(name).name != name or any(char in name for char in "*?[]"):
            raise ValueError(f"Invalid motion name for cleanup: {name}") from exc
        removed = 0
        for path in sorted(Path(output).rglob(f"{name}.*")):
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
                print(f"Removed: {path}", flush=True)
        skipped.append(name)
        print(f"Skip [{index}/{len(records)}]: {name}: {exc}; "
              f"removed {removed} output files", flush=True)
        continue
    succeeded += 1
    print(f"Wrote: {motion_path}", flush=True)
    print(f"Done [{index}/{len(records)}]: {name} ({perf_counter() - start:.2f}s)", flush=True)
print(f"Concat complete: {succeeded}/{len(records)} motions succeeded; "
      f"skipped {len(skipped)} ({perf_counter() - batch_start:.2f}s)", flush=True)
if skipped:
    print("Skipped motions: " + ", ".join(skipped), flush=True)
PY
