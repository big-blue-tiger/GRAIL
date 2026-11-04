#!/usr/bin/env python3
"""Regenerate sparse commands in a paired dataset or a separate copy."""

import argparse
from pathlib import Path
import subprocess

import joblib

from grail.place_data_generater.concat import (
    COMMAND_FIELDS, MOTION_META_FIELDS, generate_sparse_commands,
)
from grail.walk_data_tool.concat_walk_motion import load_single


def regenerate(dataset, output=None, *, lookahead_s=0.5, in_place=False):
    dataset = Path(dataset).resolve()
    output = dataset if in_place else Path(output).resolve()
    if not in_place and (dataset == output or dataset in output.parents or output in dataset.parents):
        raise ValueError("Input and output datasets must be separate")
    paths = sorted((dataset / "robot").glob("*.pkl"))
    if not paths:
        raise ValueError(f"No robot motions in {dataset}")

    if not in_place:
        # Require a fresh destination and retain the reports needed by teacher export.
        # Reflinks share disk blocks, not writable files.
        output.mkdir(parents=True, exist_ok=False)
        subprocess.run(
            ["cp", "-a", "--reflink=auto", str(dataset) + "/.", str(output)], check=True
        )
    for index, path in enumerate(paths, 1):
        name = path.stem
        key, robot = load_single(path)
        object_key, objects = load_single(dataset / "objects" / path.name)
        if key != object_key:
            raise ValueError(f"{name}: robot/object motion keys differ")
        meta = joblib.load(dataset / "meta" / path.name)
        labels, additions = generate_sparse_commands(
            robot, objects, meta, name,
            release_start_frame=meta["place_release_start_frame"], lookahead_s=lookahead_s
        )
        robot = {field: value for field, value in robot.items() if field not in COMMAND_FIELDS}
        robot.update(labels)
        meta = {field: value for field, value in meta.items() if field in MOTION_META_FIELDS}
        meta.update(additions)
        joblib.dump({key: robot}, output / "robot" / path.name)
        joblib.dump(meta, output / "meta" / path.name)

        if index % 100 == 0 or index == len(paths):
            print(f"Regenerated {index}/{len(paths)} motions", flush=True)
    return len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument("--in-place", action="store_true", help="Update the original dataset")
    parser.add_argument("--lookahead-s", type=float, default=0.5)
    args = parser.parse_args()
    regenerate(args.dataset_dir, args.output_dir, lookahead_s=args.lookahead_s, in_place=args.in_place)


if __name__ == "__main__":
    main()
