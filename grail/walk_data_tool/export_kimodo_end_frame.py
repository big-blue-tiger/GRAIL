#!/usr/bin/env python3
"""Export the first stable pose after the last foot movement as a Kimodo endpoint.

Run in the GRAIL environment; uses local Kimodo kinematics without loading a
model, CUDA, IsaacLab, or the language encoder. See README.md.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import sys
import tempfile
import types

import joblib
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot"

# Frame 0 of pickup_table_walk_concat/robot/pickup_table__alcohol_11__000.pkl.
# G1Skeleton34 local axis-angle rotations (radians), converted with
# root_quat_w_first=False and mujoco_rest_zero=False. Index 0 is the root's
# WORLD orientation, not a body joint: consumers of a local-only constraint
# must exclude it rather than constrain it to zero or to the source heading.
INITIAL_LOCAL_JOINTS_ROT = (
    (-0.020036902278661728, -0.7370942234992981, -0.025550775229930878),
    (-0.0819973275065422, 0.0, 0.0),
    (-0.17434316873550415, -0.017078252509236336, 0.19479359686374664),
    (0.0, 0.08747012913227081, 0.0),
    (0.6488083600997925, 6.818046125135681e-17, -4.165597192394567e-17),
    (-0.24743130803108215, 0.0, 0.0),
    (0.0, 0.0, -0.1620299518108368),
    (0.0, 0.0, 0.0),
    (0.04185217246413231, 0.0, 0.0),
    (-0.17485661804676056, -0.004766935016959906, 0.05437138304114342),
    (0.0, 0.027380218729376793, 0.0),
    (0.37697744369506836, 6.118895866049121e-17, -4.992389030059695e-17),
    (-0.2289874404668808, 0.0, 0.0),
    (0.0, 0.0, -0.0152793750166893),
    (0.0, 0.0, 0.0),
    (0.0, -0.036146897822618484, 0.0),
    (0.0, 0.0, 0.04240315034985542),
    (0.15839532017707825, 0.0, 0.0),
    (0.05839001387357712, -0.008401907049119473, 0.279222697019577),
    (0.0, 0.0, 0.0006410182686522603),
    (0.0, -0.4535186290740967, 0.0),
    (1.0885121822357178, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (-0.06922537088394165, -0.009536957368254662, -0.27920255064964294),
    (0.0, 0.0, 0.054997291415929794),
    (0.0, 0.5193348526954651, 0.0),
    (1.116929292678833, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)


def load_kinematics(repo: Path):
    """Expose the local package without its eager LLM/model imports.

    Only skip kimodo/__init__.py; all kinematics/conversion code and assets are
    imported unchanged from the selected checkout. This is a standalone CLI.
    """
    package = repo.resolve() / "kimodo"
    if not (package / "exports/mujoco.py").is_file():
        raise ValueError(f"Not a Kimodo checkout: {repo}")
    if "kimodo" not in sys.modules:
        module = types.ModuleType("kimodo")
        module.__path__ = [str(package)]
        module.__file__ = str(package / "__init__.py")
        sys.modules["kimodo"] = module
    elif Path(sys.modules["kimodo"].__file__).resolve().parent != package:
        raise ValueError("A different Kimodo checkout is already imported")
    from kimodo.constraints import FullBodyConstraintSet
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.geometry import matrix_to_axis_angle
    from kimodo.skeleton import G1Skeleton34

    skeleton = G1Skeleton34()
    return skeleton, MujocoQposConverter(skeleton), matrix_to_axis_angle, FullBodyConstraintSet


@dataclass(frozen=True)
class DetectionConfig:
    # Calibrated against six pickup-table video stop-time annotations.
    # Ignore small ankle/heel adjustments while retaining centimetre-scale steps.
    speed_on: float = 0.05
    speed_off: float = 0.03
    min_excursion: float = 0.03
    stable_time: float = 0.3
    smooth_window: float = 0.08

    def validate(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Detection thresholds must be finite")
        if not 0 < self.speed_off < self.speed_on:
            raise ValueError("Require 0 < foot-speed-off < foot-speed-on")
        if self.min_excursion <= 0 or self.stable_time <= 0 or self.smooth_window < 0:
            raise ValueError("Excursion/stable time must be positive; smoothing nonnegative")


def trajectory_qpos(motion):
    """Validate every frame: absolute Z-up xyz, xyzw and MuJoCo body angles."""
    fps = float(motion["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Source fps must be finite and positive")
    parts = []
    for key, width in (("root_trans_offset", 3), ("root_rot", 4), ("dof", 29)):
        value = np.asarray(motion[key])
        if value.ndim != 2 or value.shape[1] != width or len(value) == 0:
            raise ValueError(f"{key}: expected nonempty (T,{width}), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key}: nonfinite trajectory")
        parts.append(value.astype(np.float32))
    if len({len(part) for part in parts}) != 1:
        raise ValueError("Inconsistent frame counts")
    qpos = np.concatenate(parts, axis=1)
    norms = np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    if np.any(np.abs(norms - 1.0) > 0.01):
        raise ValueError("Invalid xyzw quaternion norm in trajectory")
    qpos[:, 3:7] /= norms
    return qpos


def first_qpos(motion):
    """Compatibility helper; the exporter now selects from the full trajectory."""
    return trajectory_qpos(motion)[:1]


def detect_last_step(feet_world, fps, config=DetectionConfig()):
    """Return JSON-safe selection plus filtered trajectories for diagnostics.

    Input is (T,4,3), ordered left ankle/toe, right ankle/toe, in metres.
    speed[t] measures the interval (t-1,t). A stable window starting at k
    checks the next ceil(stable_time*fps) intervals, including their endpoints.
    Its confirmation delay is NOT added to the selected source frame.
    """
    config.validate()
    feet = np.asarray(feet_world, dtype=np.float64)
    if (feet.ndim != 3 or feet.shape[1:] != (4, 3) or len(feet) == 0
            or not np.isfinite(feet).all()):
        raise ValueError("Expected finite, nonempty (T,4,3) foot trajectories")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Source fps must be finite and positive")
    # Nearest odd window, rounding an even size upwards (25 Hz -> 3 samples).
    window = max(1, int(round(config.smooth_window * fps)))
    if window % 2 == 0:
        window += 1
    filtered = median_filter(feet, size=(window, 1, 1), mode="nearest")
    point_speed = np.zeros(feet.shape[:2])
    point_speed[1:] = np.linalg.norm(np.diff(filtered, axis=0), axis=-1) * fps
    speeds = point_speed.reshape(len(feet), 2, 2).max(axis=2)
    hold = max(1, int(np.ceil(config.stable_time * fps)))
    stable = np.zeros((len(feet), 2), dtype=bool)
    if len(feet) > hold:
        # Prefix sums keep long clips linear in T, rather than T*hold.
        moving = speeds[1:] >= config.speed_off
        sums = np.concatenate([np.zeros((1, 2), dtype=int), np.cumsum(moving, axis=0)])
        stable[:len(feet) - hold] = sums[hold:] == sums[:-hold]
    episodes, ignored = [], []
    for foot, side in enumerate(("left", "right")):
        points = filtered[:, foot * 2:foot * 2 + 2]
        start = None
        excursion = 0.0
        for t in range(1, len(feet)):
            if start is None and speeds[t, foot] >= config.speed_on:
                start = t - 1
                excursion = 0.0
            if start is None:
                continue
            excursion = max(excursion, float(np.linalg.norm(points[t] - points[start], axis=-1).max()))
            if stable[t, foot]:
                event = {"foot": side, "start_frame": start, "end_frame": t,
                         "max_excursion_m": excursion, "complete": True}
                (episodes if excursion >= config.min_excursion else ignored).append(event)
                start = None
        if start is not None:
            event = {"foot": side, "start_frame": start, "end_frame": len(feet) - 1,
                     "max_excursion_m": excursion, "complete": False}
            (episodes if excursion >= config.min_excursion else ignored).append(event)
    episodes.sort(key=lambda item: (item["start_frame"], item["foot"]))
    selection = {
        "source_frame": None, "reason": None, "error": None,
        "parameters": asdict(config), "smoothing_frames": window,
        "stable_intervals": hold, "episodes": episodes, "ignored_episodes": ignored,
    }
    if any(not event["complete"] for event in episodes):
        selection["error"] = "unfinished_foot_movement"
    else:
        last_end = max((event["end_frame"] for event in episodes), default=0)
        candidates = np.flatnonzero(stable.all(axis=1) & (np.arange(len(feet)) >= last_end))
        if not len(candidates):
            selection["error"] = "no_confirmed_bilateral_stability"
        else:
            selection["source_frame"] = int(candidates[0]) if episodes else 0
            selection["stable_window_start"] = int(candidates[0])
            selection["reason"] = "after_last_step" if episodes else "no_effective_step"
    return selection, filtered, speeds


def sample_start(qpos, meta, rng):
    """Uniform in area of the 1--2 m, 150-degree annular sector.

    IsaacLab consumes table_quat as wxyz. Choose the outward normal to the
    long table edge by the robot position, never by its heading.
    """
    center = np.asarray(meta["table_pos"], dtype=float)
    size = np.asarray(meta["table_size"], dtype=float)
    quat = np.asarray(meta["table_quat"], dtype=float)
    if (center.shape != (3,) or size.shape != (3,) or quat.shape != (4,)
            or not all(np.isfinite(v).all() for v in (center, size, quat))
            or np.any(size <= 0) or abs(np.linalg.norm(quat) - 1) > 0.01):
        raise ValueError("Invalid table metadata")
    rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    if abs(rot[2, 2]) < 0.999:
        raise ValueError("Sampling requires a horizontal tabletop")
    short_axis = int(np.argmin(size[:2]))
    normal = rot[:2, short_axis]
    side = float((qpos[0, :2] - center[:2]) @ normal)
    if abs(side) <= size[short_axis] / 2:
        raise ValueError("Robot root is not outside the long table edge")
    normal = normal * np.sign(side)
    radius = float(np.sqrt(rng.uniform(1.0, 4.0)))
    angle = float(rng.uniform(-np.deg2rad(75), np.deg2rad(75)))
    tangent = np.array([-normal[1], normal[0]])
    start = qpos[0, :2] + radius * (np.cos(angle) * normal + np.sin(angle) * tangent)
    return start, radius, angle, normal


def build_constraint(args, path, key, motion, name, qpos, source_frame, kinematics):
    skeleton, converter, to_axis_angle, _ = kinematics
    destination = Path(f"{name}.json")
    server_data = args.server_data_dir
    meta_path = args.input_dir.parent / "meta" / path.name
    meta = joblib.load(meta_path)
    # Stable per-motion randomness even when --pattern changes.
    digest = hashlib.sha256(f"{args.seed}:{path.name}:{key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    start, distance, angle, normal = sample_start(qpos, meta, rng)
    duration = distance / args.speed + args.preparation_time
    num_frames = int(duration * args.fps)  # Kimodo CLI truncates.
    if num_frames < 2:
        raise ValueError("Generated motion must contain at least two frames")
    frame = num_frames - 1
    # Official complete_motion_dict needs a trajectory for smoothing
    # and velocities. Eight identical frames avoid its single-frame
    # singular system; only frame 0 is exported, without smoothing.
    converted = converter.qpos_to_motion_dict(
        np.repeat(qpos, 8, axis=0), source_fps=args.fps,
        root_quat_w_first=False, mujoco_rest_zero=False,
    )
    local = converted["local_rot_mats"][:1].clone()
    world_positions = converted["root_positions"][:1]
    # Kimodo is trained with the initial smooth root at XZ=(0,0).
    # Face the target along canonical +Z so the CLI's default
    # first_heading_angle=0 agrees with the frame-0 constraint.
    world_offset = world_positions.new_tensor([start[1], 0.0, start[0]])
    displacement = world_positions - world_offset
    heading = float(np.arctan2(displacement[0, 0].item(), displacement[0, 2].item()))
    canonical_to_world = world_positions.new_tensor(
        Rotation.from_euler("y", heading).as_matrix()
    )
    positions = displacement @ canonical_to_world
    local[:, skeleton.root_idx] = canonical_to_world.T @ local[:, skeleton.root_idx]
    constraint = {
        "type": "fullbody",
        "frame_indices": [frame],
        "local_joints_rot": to_axis_angle(local).tolist(),
        "root_positions": positions.tolist(),
    }
    # Verify the actual JSON parser and its FK, not just array shape.
    start_constraint = {
        "type": "root2d", "frame_indices": [0],
        "smooth_root_2d": [[0.0, 0.0]],
        "global_root_heading": [[1.0, 0.0]],  # [cos(0), sin(0)]: +Z
    }
    payload = json.dumps([start_constraint, constraint], indent=2, allow_nan=False) + "\n"
    from kimodo.constraints import load_constraints_lst
    parsed = load_constraints_lst(json.loads(payload), skeleton)
    loaded = parsed[1]
    if parsed[0].smooth_root_2d.abs().max().item() != 0:
        raise ValueError("Initial root constraint must be at the canonical origin")
    facing_xz = parsed[0].global_root_heading[:, [1, 0]]
    direction_xz = positions[:, [0, 2]] / positions[:, [0, 2]].norm(dim=-1, keepdim=True)
    if float((facing_xz - direction_xz).abs().max()) > 1e-5:
        raise ValueError("Initial heading must face the final root position")
    relative_distance = float(positions[0, [0, 2]].norm())
    if abs(relative_distance - distance) > 1e-5:
        raise ValueError("Constraint displacement differs from sampled walking distance")
    fk_error = float((loaded.global_joints_positions @ canonical_to_world.T + world_offset
                      - converted["posed_joints"][:1]).abs().max())
    if fk_error > 1e-5:
        raise ValueError(f"JSON FK error {fk_error} exceeds 1e-5 m")
    roundtrip = converter.dict_to_qpos(converted, root_quat_w_first=False)
    dof_error = float(np.max(np.abs(roundtrip[0, 7:] - qpos[0, 7:])))
    command = shlex.join([
        "python", "-m", "kimodo.scripts.generate",
        "A person walks forward slowly.",
        "--model", "Kimodo-G1-RP-v1", "--duration", repr(duration),
        "--constraints", str(server_data / destination.name),
        "--num_samples", "1", "--seed", str(args.seed),
        "--output", str(server_data / "motions" / name),
    ])
    record = {
        "start_xy_isaaclab": start.tolist(),
        "end_xy_isaaclab": qpos[0, :2].tolist(),
        "sector_normal_xy_isaaclab": normal.tolist(),
        "table_meta": str(meta_path.resolve()),
        "distance_m": distance, "sector_angle_deg": float(np.rad2deg(angle)),
        "generation_duration": duration, "num_frames": num_frames,
        "target_frame": frame, "generation_command": command,
        "source": str(path.resolve()), "motion_key": str(key),
        "source_frame": source_frame, "source_fps": float(motion["fps"]),
        "constraint": destination.name, "root_positions": positions.tolist(),
        "world_root_positions": world_positions.tolist(),
        "first_heading_angle": 0.0,
        "start_heading_angle_world_kimodo": heading,
        "canonical_to_world_rotation_kimodo": canonical_to_world.tolist(),
        "canonical_to_world_rotation_isaaclab": canonical_to_world[[2, 0, 1]][:, [2, 0, 1]].tolist(),
        "canonical_to_world_translation_kimodo": world_offset.tolist(),
        "canonical_to_world_translation_isaaclab": [float(start[0]), float(start[1]), 0.0],
        "json_fk_max_error_m": fk_error,
        "official_csv_roundtrip_max_dof_error_rad": dof_error,
    }
    return payload, record


_KINEMATICS = None
_WORKER_ARGS = None


def init_worker(args):
    global _KINEMATICS, _WORKER_ARGS
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    torch.set_num_threads(1)
    _WORKER_ARGS = args
    _KINEMATICS = load_kinematics(args.kimodo_root)


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def json_text(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def process_file(task):
    """One worker owns each PKL and its uniquely preflighted diagnostic paths."""
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from grail.walk_data_tool.foot_step_diagnostics import write_plot, write_replay

    path, expected_hash, entries = task
    args = _WORKER_ARGS
    results = []
    try:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("Source changed after preflight")
        motions = joblib.load(io.BytesIO(data))
    except Exception as exc:
        motions = None
        load_error = str(exc)
    for key, name in entries:
        selection = None
        feet = filtered = speeds = positions = None
        record = payload = error = None
        fps = None
        diagnostics = {"plot": f"diagnostics/{name}.png"}
        try:
            if motions is None:
                raise ValueError(load_error)
            motion = motions[key]
            qpos = trajectory_qpos(motion)
            fps = float(motion["fps"])
            skeleton, converter, _, _ = _KINEMATICS
            # Pad only the FK converter's derived smoothing channels on short
            # clips; padded frames never participate in foot-step detection.
            padded = np.concatenate([qpos, np.repeat(qpos[-1:], max(0, 8 - len(qpos)), axis=0)])
            with torch.inference_mode():
                converted = converter.qpos_to_motion_dict(
                    padded, source_fps=fps, root_quat_w_first=False, mujoco_rest_zero=False)
                positions = converted["posed_joints"][:len(qpos)].cpu().numpy()[..., [2, 0, 1]]
                feet = positions[:, skeleton.foot_joint_idx]
                selection, filtered, speeds = detect_last_step(feet, fps, args.detection)
                if selection["error"]:
                    raise ValueError(selection["error"])
                k = selection["source_frame"]
                if len(qpos) - k <= args.transition_frames:
                    raise ValueError(f"Selected frame {k} leaves {len(qpos) - k} frames; "
                                     f"PCHIP requires at least {args.transition_frames + 1}")
                payload, record = build_constraint(
                    args, path, key, motion, name, qpos[k:k + 1], k, _KINEMATICS)
            record.update({"source_sha256": expected_hash, "source_num_frames": len(qpos),
                           "selection": selection, "diagnostics": diagnostics})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        try:
            write_plot(args.output_dir / diagnostics["plot"], name, feet, filtered,
                       speeds, fps, selection, error)
            if args.replay and positions is not None:
                replay = f"diagnostics/{name}.html"
                write_replay(args.output_dir / replay, name, positions, fps,
                             _KINEMATICS[0], selection, error)
                diagnostics["replay"] = replay
        except Exception as exc:
            error = f"{error + '; ' if error else ''}diagnostics: {type(exc).__name__}: {exc}"
        results.append({
            "source": str(path), "motion_key": str(key), "name": name,
            "payload": payload if error is None else None,
            "record": record if error is None else None,
            "error": error, "selection": selection, "diagnostics": diagnostics,
        })
    return results


def preflight(files, args):
    """Reserve all paths before starting workers; bad files remain batch failures."""
    tasks, failures = [], []
    paths = {args.output_dir / p for p in ("manifest.json", "generate_commands.sh", "failures.json")}
    for path in files:
        try:
            data = path.read_bytes()
            motions = joblib.load(io.BytesIO(data))
            if not isinstance(motions, dict) or not motions:
                raise ValueError("Expected a nonempty motion-key dictionary")
            if len({str(key) for key in motions}) != len(motions):
                raise ValueError("Motion keys must have unique string representations")
            entries = [(key, path.stem if len(motions) == 1 else f"{path.stem}__motion_{i:03d}")
                       for i, key in enumerate(motions)]
            source_hash = hashlib.sha256(data).hexdigest()
        except Exception as exc:
            entries = [(None, path.stem)]
            failures.append({"source": str(path), "motion_key": None, "name": path.stem,
                             "error": f"{type(exc).__name__}: {exc}",
                             "diagnostics": {"plot": f"diagnostics/{path.stem}.png"}})
            source_hash = None
        for _, name in entries:
            for relative in (f"{name}.json", f"diagnostics/{name}.png", f"diagnostics/{name}.html"):
                destination = args.output_dir / relative
                if destination in paths:
                    raise ValueError(f"Output name collision: {destination}")
                paths.add(destination)
        if source_hash is not None:
            tasks.append((path, source_hash, entries))
    for destination in sorted(paths):
        if destination.is_symlink() or destination.resolve() != destination:
            raise ValueError(f"Refusing symlink output: {destination}")
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"{destination} exists; use --overwrite to replace")
    return tasks, failures


def export(args):
    if not all(np.isfinite(v) for v in (args.fps, args.speed, args.preparation_time)):
        raise ValueError("fps, speed and preparation-time must be finite")
    if args.fps <= 0 or args.speed <= 0 or args.preparation_time < 0:
        raise ValueError("fps/speed must be positive; preparation-time must be nonnegative")
    if args.workers < 1 or args.transition_frames < 1:
        raise ValueError("workers and transition-frames must be positive")
    args.detection.validate()
    args.input_dir = args.input_dir.resolve()
    requested_output = args.output_dir or args.input_dir.parent / "kimodo_end_frame"
    if requested_output.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    args.output_dir = requested_output.resolve()
    if args.output_dir == args.input_dir or args.input_dir in args.output_dir.parents:
        raise ValueError("Output must be outside the source robot directory")
    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise ValueError(f"No PKL files matched in {args.input_dir}")
    tasks, failures = preflight(files, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from grail.walk_data_tool.foot_step_diagnostics import write_plot

    for failure in failures:
        write_plot(args.output_dir / failure["diagnostics"]["plot"], failure["name"],
                   None, None, None, None, None, failure["error"])
        if args.overwrite:
            (args.output_dir / f"{failure['name']}.json").unlink(missing_ok=True)
            (args.output_dir / f"diagnostics/{failure['name']}.html").unlink(missing_ok=True)
    commands = [
        "#!/usr/bin/env bash", "set -euo pipefail",
        shlex.join(["cd", str(args.server_kimodo_root)]),
        shlex.join(["source", str(args.server_kimodo_root / "deployment/env.sh")]),
        "export HF_HUB_OFFLINE=1", "export TEXT_ENCODER_MODE=local",
        "export TEXT_ENCODER_DTYPE=float32",
        shlex.join(["mkdir", "-p", str(args.server_data_dir / "motions")]), "",
    ]
    records = []
    workers = min(args.workers, len(tasks))
    # Inherited by spawned imports, before numpy/scipy import their BLAS runtime.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    pool = None
    try:
        if workers > 1:
            pool = ProcessPoolExecutor(max_workers=workers,
                                       mp_context=multiprocessing.get_context("spawn"),
                                       initializer=init_worker, initargs=(args,))
            batches = pool.map(process_file, tasks)
        elif tasks:
            init_worker(args)
            batches = map(process_file, tasks)
        else:
            batches = []
        for i, batch in enumerate(batches, 1):
            for result in batch:
                destination = args.output_dir / f"{result['name']}.json"
                if result["error"] is None:
                    atomic_text(destination, result["payload"])
                    records.append(result["record"])
                    commands.append(result["record"]["generation_command"])
                else:
                    if args.overwrite:
                        destination.unlink(missing_ok=True)
                    failures.append({k: v for k, v in result.items() if k not in {"payload", "record"}})
                    print(f"FAILED {result['name']}: {result['error']}", file=sys.stderr, flush=True)
                if args.overwrite and "replay" not in result["diagnostics"]:
                    (args.output_dir / f"diagnostics/{result['name']}.html").unlink(missing_ok=True)
            print(f"[{i}/{len(tasks)}] {len(records)} exported, {len(failures)} failed", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    failures.sort(key=lambda item: (item["source"], str(item["motion_key"])))
    manifest = {
        "schema_version": 2,
        "kimodo_root": str(args.kimodo_root.resolve()), "skeleton": "g1skel34",
        "generation_fps": args.fps, "speed_m_s": args.speed,
        "preparation_time_s": args.preparation_time, "seed": args.seed,
        "detection": asdict(args.detection), "transition_frames": args.transition_frames,
        "sampling": "uniform area; radii 1..2 m; angle -75..75 deg; one start per motion",
        "coordinate_policy": "Y-up; initial smooth root XZ zero; face target along +Z; world = canonical @ rotation.T + translation",
        "records": records, "failures": failures,
        "summary": {"files": len(files), "succeeded": len(records), "failed": len(failures)},
    }
    atomic_text(args.output_dir / "manifest.json", json_text(manifest))
    atomic_text(args.output_dir / "failures.json", json_text(failures))
    atomic_text(args.output_dir / "generate_commands.sh", "\n".join(commands) + "\n")
    print(f"Exported {len(records)} poses from {len(files)} PKLs to {args.output_dir}; "
          f"{len(failures)} failed", flush=True)
    if records:
        print(f"Max JSON FK error: {max(r['json_fk_max_error_m'] for r in records):.3g} m")
    return manifest


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    detection_defaults = DetectionConfig()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, help="Default: input-dir/../kimodo_end_frame")
    parser.add_argument("--kimodo-root", type=Path, default=ROOT.parent / "kimodo")
    parser.add_argument(
        "--server-kimodo-root", type=Path, default=Path("/workspace/kimodo"),
        help="Kimodo checkout on the server, used in generate_commands.sh",
    )
    parser.add_argument(
        "--server-data-dir", type=Path, default=Path("/workspace/kimodo/data/local_pickup"),
        help="Server directory containing uploaded constraint JSONs; motions are written below it",
    )
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--speed", type=float, default=0.5, help="Nominal walking speed, m/s")
    parser.add_argument("--preparation-time", type=float, default=0.1,
                        help="Extra duration in seconds (default: 0.1); does not impose a stationary hold")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible per-motion sampling seed")
    parser.add_argument("--fps", type=float, default=30.0, help="Kimodo model FPS, not PKL FPS")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="CPU processes (default: up to 4); 1 for serial")
    parser.add_argument("--foot-speed-on", type=float, default=detection_defaults.speed_on,
                        help="Movement onset, m/s")
    parser.add_argument("--foot-speed-off", type=float, default=detection_defaults.speed_off,
                        help="Stable speed ceiling, m/s")
    parser.add_argument("--min-foot-excursion", type=float, default=detection_defaults.min_excursion,
                        help="Minimum movement, metres")
    parser.add_argument("--stable-time", type=float, default=detection_defaults.stable_time,
                        help="Required stable duration, seconds")
    parser.add_argument("--smooth-window", type=float, default=detection_defaults.smooth_window,
                        help="Median window, seconds; 0 disables")
    parser.add_argument("--transition-frames", type=int, default=10,
                        help="Reserve at least N+1 source frames for downstream PCHIP")
    parser.add_argument("--replay", action="store_true", help="Write offline skeleton HTML around selected frame")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.detection = DetectionConfig(args.foot_speed_on, args.foot_speed_off,
                                     args.min_foot_excursion, args.stable_time, args.smooth_window)
    try:
        manifest = export(args)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    if manifest["failures"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
