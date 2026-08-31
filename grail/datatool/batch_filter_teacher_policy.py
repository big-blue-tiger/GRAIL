#!/usr/bin/env python3
"""Filter a motion library by running a standalone SONIC teacher policy.

The teacher checkpoint is passed directly to ``gear_sonic/eval_agent_trl.py``
as its primary checkpoint. Motions that reach only the normal ``time_out``
termination are copied verbatim to a new motion library; motions that trigger
any other termination term are rejected.

Example::

    python -u -m grail.datatool.batch_filter_teacher_policy \
        --data_dir data/hf_dataset/data_update/data/pickup_table \
        --output_dir data/hf_dataset/data_update/data/pickup_table_teacher_cleaned \
        --teacher_checkpoint imports/SONIC/models/pnp_table/last.pt \
        --num_envs 24 \
        --no_record_video
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPORT_FORMAT_VERSION = 1


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON through a temporary sibling and atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy one file without exposing a partially written destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def find_teacher_config(checkpoint: Path) -> Path:
    """Resolve the config location using the evaluator's two supported layouts."""
    candidates = (checkpoint.parent / "config.yaml", checkpoint.parent.parent / "config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(
        "teacher checkpoint has no config.yaml beside it or in its parent directory: "
        f"{checkpoint}"
    )


def chunked(values: list[str], size: int) -> list[list[str]]:
    """Split values into stable, non-empty chunks."""
    if size <= 0:
        raise ValueError("num_envs must be positive")
    return [values[start : start + size] for start in range(0, len(values), size)]


def hydra_list(values: list[str]) -> str:
    """Serialize strings as a compact Hydra-compatible list."""
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def build_eval_command(
    *,
    python_executable: str,
    sonic_root: Path,
    teacher_checkpoint: Path,
    data_dir: Path,
    asset_root: Path,
    motion_keys: list[str],
    report_path: Path,
    hydra_dir: Path,
    render_dir: Path,
    headless: bool,
    record_video: bool,
) -> list[str]:
    """Construct one evaluator command without any student-policy arguments."""
    batch_size = len(motion_keys)
    if batch_size == 0:
        raise ValueError("cannot build an eval command for an empty batch")
    filter_value = hydra_list(motion_keys)
    command = [
        python_executable,
        "-u",
        str(sonic_root / "gear_sonic" / "eval_agent_trl.py"),
        f"+checkpoint={teacher_checkpoint}",
        f"+headless={str(headless)}",
        "+run_once=True",
        "++run_eval_loop=True",
        "++use_wandb=False",
        "++motion_shard_by_rank=False",
        "++load_only_num_envs_motions=True",
        f"++num_envs={batch_size}",
        f"++run_once_report_path={report_path}",
        f"hydra.run.dir={hydra_dir}",
        f"++manager_env.config.object_usd_path={data_dir / 'object_usd'}",
        f"++manager_env.commands.motion.filter_motion_keys={filter_value}",
        f"++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys={filter_value}",
        f"++manager_env.commands.motion.motion_lib_cfg.max_unique_motions={batch_size}",
        "++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0",
        "++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={data_dir / 'robot'}",
        f"++manager_env.commands.motion.motion_lib_cfg.object_motion_file={data_dir / 'objects'}",
        f"++manager_env.commands.motion.motion_lib_cfg.bps_dir={data_dir / 'bps'}",
        f"++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot={asset_root}",
    ]
    if record_video:
        command.extend(
            [
                "++manager_env.config.render_results=True",
                f"++manager_env.config.save_rendering_dir={render_dir}",
                f"++manager_env.config.max_render_envs={batch_size}",
                "~manager_env/recorders=empty",
                "+manager_env/recorders=render",
            ]
        )
    else:
        command.append("++manager_env.config.render_results=False")
    return command


def validate_motion_input(data_dir: Path, motion_key: str) -> None:
    """Check that all assets needed by teacher eval and clean export exist."""
    required = (
        data_dir / "robot" / f"{motion_key}.pkl",
        data_dir / "objects" / f"{motion_key}.pkl",
        data_dir / "meta" / f"{motion_key}.pkl",
        data_dir / "bps" / f"{motion_key}.npy",
    )
    missing = [str(path) for path in required if not path.is_file()]
    usd_candidates = (
        data_dir / "object_usd" / f"{motion_key}.usd",
        data_dir / "object_usd" / f"{motion_key}.usda",
    )
    if not any(path.is_file() for path in usd_candidates):
        missing.append(" or ".join(str(path) for path in usd_candidates))
    if missing:
        raise ValueError(f"{motion_key}: missing required input: " + ", ".join(missing))


def validate_eval_report(
    report_path: Path, expected_keys: list[str], teacher_checkpoint: Path
) -> list[dict]:
    """Load a complete evaluator report and enforce exact batch correspondence."""
    if not report_path.is_file():
        raise ValueError(f"evaluator did not produce report: {report_path}")
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)
    if report.get("format_version") != 1:
        raise ValueError("unsupported evaluator report format")
    if not report.get("complete", False):
        raise ValueError("evaluator report is incomplete")
    reported_checkpoint = Path(report.get("checkpoint", "")).resolve()
    if reported_checkpoint != teacher_checkpoint.resolve():
        raise ValueError(
            f"evaluator used checkpoint {reported_checkpoint}, expected {teacher_checkpoint}"
        )
    results = report.get("results", [])
    if len(results) != len(expected_keys):
        raise ValueError(
            f"evaluator returned {len(results)} motions, expected {len(expected_keys)}"
        )
    result_by_key = {}
    for result in results:
        key = result.get("motion_key")
        if not isinstance(key, str) or key in result_by_key:
            raise ValueError(f"invalid or duplicate motion key in evaluator report: {key!r}")
        if result.get("status") not in {"accepted", "early_terminated"}:
            raise ValueError(f"invalid evaluator status for {key}: {result.get('status')!r}")
        result_by_key[key] = result
    if set(result_by_key) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(result_by_key))
        unexpected = sorted(set(result_by_key) - set(expected_keys))
        raise ValueError(
            f"evaluator motion mismatch; missing={missing}, unexpected={unexpected}"
        )
    return [result_by_key[key] for key in expected_keys]


def copy_shared_assets(data_dir: Path, output_dir: Path) -> None:
    """Copy shared files that are not named after an individual motion."""
    shared_files = (
        (data_dir / "bps" / "_basis.npy", output_dir / "bps" / "_basis.npy"),
        (
            data_dir / "object_usd" / "config.yaml",
            output_dir / "object_usd" / "config.yaml",
        ),
    )
    for source, destination in shared_files:
        if source.is_file():
            atomic_copy(source, destination)


def remove_exported_motion(output_dir: Path, motion_key: str) -> None:
    """Remove files for one rejected or partially exported motion."""
    candidates = (
        output_dir / "robot" / f"{motion_key}.pkl",
        output_dir / "objects" / f"{motion_key}.pkl",
        output_dir / "meta" / f"{motion_key}.pkl",
        output_dir / "bps" / f"{motion_key}.npy",
        output_dir / "object_usd" / f"{motion_key}.usd",
        output_dir / "object_usd" / f"{motion_key}.usda",
    )
    for path in candidates:
        path.unlink(missing_ok=True)

    texture_root = output_dir / "object_usd" / "textures"
    nested_textures = texture_root / motion_key
    if nested_textures.is_dir():
        shutil.rmtree(nested_textures)
    for path in texture_root.glob(f"{motion_key}_*") if texture_root.is_dir() else ():
        if path.is_file():
            path.unlink()


def summarize_report(report: dict) -> None:
    """Refresh aggregate counts after a motion-level update."""
    counts = {"pending": 0, "accepted": 0, "rejected": 0, "eval_error": 0}
    for entry in report.get("motions", {}).values():
        status = entry.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    report["summary"] = counts
    report["updated_at_unix"] = time.time()


def save_clean_report(report: dict, report_path: Path) -> None:
    summarize_report(report)
    atomic_write_json(report_path, report)


def load_or_create_clean_report(
    *,
    report_path: Path,
    data_dir: Path,
    output_dir: Path,
    teacher_checkpoint: Path,
    teacher_config: Path,
    num_envs: int,
    motion_keys: list[str],
    resume: bool,
) -> dict:
    """Create a cleaning report or validate one used for resume."""
    if resume:
        if not report_path.is_file():
            raise ValueError("--resume requires an existing clean_report.json")
        with report_path.open(encoding="utf-8") as file:
            report = json.load(file)
        if report.get("format_version") != REPORT_FORMAT_VERSION:
            raise ValueError("resume report format_version is not supported")
        expected = {
            "source_data_dir": str(data_dir),
            "output_data_dir": str(output_dir),
            "teacher_checkpoint": str(teacher_checkpoint),
        }
        for key, value in expected.items():
            if report.get(key) != value:
                raise ValueError(f"resume report {key} does not match: {report.get(key)!r}")
        if int(report.get("num_envs", -1)) != num_envs:
            raise ValueError("resume report num_envs does not match --num_envs")
        if report.get("selected_motion_keys") != motion_keys:
            raise ValueError("resume motion selection differs from the existing report")
    else:
        report = {
            "format_version": REPORT_FORMAT_VERSION,
            "source_data_dir": str(data_dir),
            "output_data_dir": str(output_dir),
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_config": str(teacher_config),
            "num_envs": int(num_envs),
            "selected_motion_keys": motion_keys,
            "created_at_unix": time.time(),
            "motions": {},
            "batches": [],
        }
    for motion_key in motion_keys:
        report.setdefault("motions", {}).setdefault(
            motion_key, {"status": "pending", "reason_code": "not_evaluated"}
        )
    save_clean_report(report, report_path)
    return report


def select_motion_keys(data_dir: Path, requested: str | None, maximum: int) -> list[str]:
    """Select stable motion keys from the robot directory."""
    robot_dir = data_dir / "robot"
    if not robot_dir.is_dir():
        raise ValueError(f"input robot directory does not exist: {robot_dir}")
    available = sorted(
        path.stem
        for path in robot_dir.glob("*.pkl")
        if not path.name.endswith(".trajectory.pkl")
    )
    if requested:
        keys = [key.strip() for key in requested.split(",") if key.strip()]
        unknown = sorted(set(keys) - set(available))
        if unknown:
            raise ValueError("unknown motion keys: " + ", ".join(unknown))
        if len(keys) != len(set(keys)):
            raise ValueError("--motion_keys contains duplicates")
    else:
        keys = available
    if maximum > 0:
        keys = keys[:maximum]
    if not keys:
        raise ValueError("no motions selected")
    return keys


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    sonic_root = repo_root / "imports" / "SONIC"
    parser = argparse.ArgumentParser(
        description="Filter a motion library using a standalone teacher policy"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=24)
    parser.add_argument("--sonic_root", default=str(sonic_root))
    parser.add_argument(
        "--asset_root",
        default=None,
        help=(
            "MJCF asset root (default: "
            "<sonic_root>/gear_sonic/data/assets/robot_description/mjcf)"
        ),
    )
    parser.add_argument("--motion_keys", default=None, help="Comma-separated motion keys")
    parser.add_argument("--max_motions", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--loguru_level", default="INFO")
    headless_group = parser.add_mutually_exclusive_group()
    headless_group.add_argument("--headless", dest="headless", action="store_true")
    headless_group.add_argument("--no_headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument("--record_video", dest="record_video", action="store_true")
    video_group.add_argument("--no_record_video", dest="record_video", action="store_false")
    parser.set_defaults(record_video=False)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.max_motions < 0:
        raise ValueError("--max_motions must be non-negative")

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    teacher_checkpoint = Path(args.teacher_checkpoint).resolve()
    sonic_root = Path(args.sonic_root).resolve()
    asset_root = (
        Path(args.asset_root).resolve()
        if args.asset_root
        else sonic_root / "gear_sonic" / "data" / "assets" / "robot_description" / "mjcf"
    )
    if data_dir == output_dir:
        raise ValueError("--output_dir must differ from --data_dir")
    if not teacher_checkpoint.is_file():
        raise ValueError(f"teacher checkpoint does not exist: {teacher_checkpoint}")
    teacher_config = find_teacher_config(teacher_checkpoint)
    eval_script = sonic_root / "gear_sonic" / "eval_agent_trl.py"
    if not eval_script.is_file():
        raise ValueError(f"eval_agent_trl.py does not exist under --sonic_root: {eval_script}")
    if not asset_root.is_dir():
        raise ValueError(f"asset root does not exist: {asset_root}")

    output_nonempty = output_dir.is_dir() and any(output_dir.iterdir())
    if output_nonempty and not args.resume:
        raise ValueError("--output_dir is non-empty; use a new directory or pass --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "clean_report.json"
    work_dir = output_dir / ".teacher_policy_work"

    motion_keys = select_motion_keys(data_dir, args.motion_keys, args.max_motions)
    report = load_or_create_clean_report(
        report_path=report_path,
        data_dir=data_dir,
        output_dir=output_dir,
        teacher_checkpoint=teacher_checkpoint,
        teacher_config=teacher_config,
        num_envs=args.num_envs,
        motion_keys=motion_keys,
        resume=args.resume,
    )

    pending_keys = []
    for motion_key in motion_keys:
        previous = report["motions"][motion_key]
        if args.resume and previous.get("status") == "accepted":
            try:
                from grail.datatool.batch_render_replay_clip import verify_exported_pair

                verify_exported_pair(str(output_dir), motion_key)
                continue
            except Exception as exc:  # noqa: BLE001
                previous = {
                    "status": "pending",
                    "reason_code": "accepted_export_needs_repair",
                    "error": str(exc),
                }
                report["motions"][motion_key] = previous
        elif args.resume and previous.get("status") == "rejected":
            remove_exported_motion(output_dir, motion_key)
            continue
        try:
            validate_motion_input(data_dir, motion_key)
            report["motions"][motion_key] = {
                "status": "pending",
                "reason_code": "preflight_passed",
            }
            pending_keys.append(motion_key)
        except Exception as exc:  # noqa: BLE001
            report["motions"][motion_key] = {
                "status": "eval_error",
                "reason_code": "preflight_error",
                "error": str(exc),
            }
        save_clean_report(report, report_path)

    print(f"Teacher checkpoint: {teacher_checkpoint}")
    print(f"Teacher config: {teacher_config}")
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Selected: {len(motion_keys)}, pending evaluation: {len(pending_keys)}")
    if args.dry_run:
        for batch_index, batch_keys in enumerate(chunked(pending_keys, args.num_envs)):
            batch_dir = work_dir / f"batch_{batch_index:04d}"
            command = build_eval_command(
                python_executable=sys.executable,
                sonic_root=sonic_root,
                teacher_checkpoint=teacher_checkpoint,
                data_dir=data_dir,
                asset_root=asset_root,
                motion_keys=batch_keys,
                report_path=batch_dir / "run_once_report.json",
                hydra_dir=batch_dir / "hydra",
                render_dir=batch_dir / "renderings",
                headless=args.headless,
                record_video=args.record_video,
            )
            print(f"DRY RUN batch {batch_index}: " + " ".join(command))
        return int(
            any(
                report["motions"][key].get("status") == "eval_error"
                for key in motion_keys
            )
        )

    had_error = any(
        report["motions"][key].get("status") == "eval_error" for key in motion_keys
    )
    for batch_index, batch_keys in enumerate(chunked(pending_keys, args.num_envs)):
        batch_dir = work_dir / f"batch_{batch_index:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_report_path = batch_dir / "run_once_report.json"
        # A resumed run may reuse a batch directory. Never allow a stale
        # complete report to make a crashed evaluator look successful.
        batch_report_path.unlink(missing_ok=True)
        command = build_eval_command(
            python_executable=sys.executable,
            sonic_root=sonic_root,
            teacher_checkpoint=teacher_checkpoint,
            data_dir=data_dir,
            asset_root=asset_root,
            motion_keys=batch_keys,
            report_path=batch_report_path,
            hydra_dir=batch_dir / "hydra",
            render_dir=batch_dir / "renderings",
            headless=args.headless,
            record_video=args.record_video,
        )
        print(
            f"Evaluating batch {batch_index + 1}/{len(chunked(pending_keys, args.num_envs))} "
            f"({len(batch_keys)} motions)",
            flush=True,
        )
        log_path = batch_dir / "eval.log"
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["LOGURU_LEVEL"] = args.loguru_level
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=sonic_root,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )

        batch_record = {
            "batch_index": batch_index,
            "motion_keys": batch_keys,
            "returncode": process.returncode,
            "eval_report": str(batch_report_path),
            "log": str(log_path),
        }
        try:
            if process.returncode != 0:
                raise RuntimeError(f"teacher evaluator exited with code {process.returncode}")
            eval_results = validate_eval_report(
                batch_report_path, batch_keys, teacher_checkpoint
            )
        except Exception as exc:  # noqa: BLE001
            had_error = True
            batch_record["status"] = "eval_error"
            batch_record["error"] = str(exc)
            for motion_key in batch_keys:
                report["motions"][motion_key] = {
                    "status": "eval_error",
                    "reason_code": "teacher_eval_failed",
                    "error": str(exc),
                    "batch_index": batch_index,
                }
            report.setdefault("batches", []).append(batch_record)
            save_clean_report(report, report_path)
            print(f"Batch {batch_index} failed: {exc}; log={log_path}", file=sys.stderr)
            continue

        batch_record["status"] = "complete"
        for eval_result in eval_results:
            motion_key = eval_result["motion_key"]
            if eval_result["status"] == "early_terminated":
                remove_exported_motion(output_dir, motion_key)
                report["motions"][motion_key] = {
                    "status": "rejected",
                    "reason_code": "teacher_early_termination",
                    "batch_index": batch_index,
                    "episode_steps": eval_result["episode_steps"],
                    "elapsed_seconds": eval_result["elapsed_seconds"],
                    "termination_reasons": eval_result["termination_reasons"],
                    "timed_out": eval_result["timed_out"],
                }
                save_clean_report(report, report_path)
                continue
            try:
                from grail.datatool.batch_render_replay_clip import export_accepted_motion

                output_frames = export_accepted_motion(
                    str(data_dir), str(output_dir), motion_key, crop_start_frame=0
                )
                report["motions"][motion_key] = {
                    "status": "accepted",
                    "reason_code": "teacher_time_out",
                    "batch_index": batch_index,
                    "episode_steps": eval_result["episode_steps"],
                    "elapsed_seconds": eval_result["elapsed_seconds"],
                    "termination_reasons": eval_result["termination_reasons"],
                    "timed_out": True,
                    "output_frames": output_frames,
                }
            except Exception as exc:  # noqa: BLE001
                remove_exported_motion(output_dir, motion_key)
                had_error = True
                report["motions"][motion_key] = {
                    "status": "eval_error",
                    "reason_code": "export_failed",
                    "error": str(exc),
                    "batch_index": batch_index,
                }
            save_clean_report(report, report_path)
        report.setdefault("batches", []).append(batch_record)
        save_clean_report(report, report_path)

    if report.get("summary", {}).get("accepted", 0) > 0:
        copy_shared_assets(data_dir, output_dir)
    save_clean_report(report, report_path)
    summary = report["summary"]
    print(
        "Complete: "
        f"accepted={summary.get('accepted', 0)}, rejected={summary.get('rejected', 0)}, "
        f"errors={summary.get('eval_error', 0)}"
    )
    print(f"Report: {report_path}")
    return 1 if had_error or summary.get("eval_error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
