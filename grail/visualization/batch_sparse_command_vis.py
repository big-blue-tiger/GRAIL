#!/usr/bin/env python3
"""Annotate each completed replay with sparse commands, scene metadata and live object xyz."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import warnings

import joblib
import numpy as np


REFRESH = Fraction(3, 10)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SceneContext:
    meta: dict = field(default_factory=dict)
    object_positions: np.ndarray | None = None


def load_commands(path):
    motions = joblib.load(path)
    if len(motions) != 1:
        raise ValueError(f"{path}: expected one motion")
    robot = next(iter(motions.values()))
    fps = float(robot["fps"])
    total = len(robot["root_trans_offset"])
    if total == 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: invalid frame count or fps")
    labels = []
    for field, channels in (("policy_command_x_yaw", 2), ("policy_command_drop_button", 1)):
        if field not in robot:
            raise ValueError(f"{path}: missing {field}; generate sparse commands first")
        value = np.asarray(robot[field])
        if value.shape != (total, channels) or not np.isfinite(value).all():
            raise ValueError(f"{path}: invalid {field} shape or values")
        labels.append(value)
    return fps, labels[0], labels[1]


def load_scene_context(robot_path, fps, total):
    """Pair by filename, since the motion key inside a PKL can differ from its stem."""
    robot_path = Path(robot_path)
    dataset = robot_path.parent.parent
    meta_path = dataset / "meta" / robot_path.name
    object_path = dataset / "objects" / robot_path.name
    meta = joblib.load(meta_path) if meta_path.is_file() else {}
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path}: expected a metadata dictionary")
    positions = None
    if object_path.is_file():
        motions = joblib.load(object_path)
        if len(motions) != 1:
            raise ValueError(f"{object_path}: expected one motion")
        objects = next(iter(motions.values()))
        object_fps = float(objects.get("fps", fps))
        if not math.isfinite(object_fps) or not math.isclose(object_fps, fps, abs_tol=1e-5):
            raise ValueError(f"{object_path}: object fps {object_fps} differs from robot fps {fps}")
        positions = np.asarray(objects["root_pos"])
        if positions.ndim == 3 and positions.shape[1] >= 1:
            positions = positions[:, 0, :]
        if positions.shape != (total, 3) or not np.isfinite(positions).all():
            raise ValueError(f"{object_path}: expected finite root_pos with {total} frames and xyz")
    missing = [str(p) for p in (meta_path, object_path) if not p.is_file()]
    if missing:
        warnings.warn(f"Missing scene context (shown as N/A): {', '.join(missing)}", stacklevel=2)
    return SceneContext(meta, positions)


def probe_video(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    if stream.get("nb_frames", "N/A") == "N/A":
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        )
        stream["nb_frames"] = json.loads(result.stdout)["streams"][0]["nb_read_frames"]
    return stream


def validate_timeline(stream, fps, total):
    video_fps = float(Fraction(stream["avg_frame_rate"]))
    if int(stream["nb_frames"]) != total or not math.isclose(video_fps, fps, abs_tol=1e-5):
        raise ValueError(
            f"Video/data timeline mismatch: video {stream['nb_frames']} frames at {video_fps} fps, "
            f"data {total} frames at {fps} fps. Use an untrimmed replay starting at frame 0."
        )


def ass_time(centiseconds):
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def format_vector(value, size=3):
    if value is None:
        return "N/A"
    values = np.asarray(value, dtype=float)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite {size}-component vector, got {value!r}")
    return "[" + ", ".join(f"{0.0 if abs(v) < 0.0000005 else v:.6f}" for v in values) + "]"


def meta_frame(meta, key):
    value = meta.get(key)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(f"Invalid meta.{key}: {value!r}")
    return int(number)


def scene_lines(context):
    meta, positions = context.meta, context.object_positions
    initial = format_vector(positions[0] if positions is not None else None)
    final = format_vector(positions[-1] if positions is not None else None)
    final_frame = len(positions) - 1 if positions is not None else "N/A"
    carry_frame = meta_frame(meta, "place_carry_source_frame")
    release_start = meta_frame(meta, "place_release_start_frame")
    release_frame = release_start - 1 if release_start is not None and release_start > 0 else "N/A"

    def frame(key):
        value = meta_frame(meta, key)
        return value if value is not None else "N/A"

    return [
        f"Object initial [world xyz,m], frame=0: {initial}",
        f"Carry goal [world xyz,m], place_carry_source_frame={carry_frame if carry_frame is not None else 'N/A'}:",
        f"  place_object_carry_goal_pos_w={format_vector(meta.get('place_object_carry_goal_pos_w'))}",
        f"Release goal [world xyz,m], frame={release_frame} (place_release_start_frame - 1):",
        f"  place_object_goal_pos_w={format_vector(meta.get('place_object_goal_pos_w'))}",
        f"Object final [world xyz,m], frame={final_frame}: {final}",
        f"transition_start_frame={frame('transition_start_frame')}: teacher latent residual fade-in begins",
        f"transition_end_frame={frame('transition_end_frame')}: full teacher residual weight (NOT GRAIL end)",
        f"grail_start_frame={frame('grail_start_frame')}: configured body tracking rewards masked",
        f"place_generated_start_frame={frame('place_generated_start_frame')}: those body tracking rewards restored",
        f"table_pos [world xyz,m]={format_vector(meta.get('table_pos'))}",
        f"table_size [xyz,m]={format_vector(meta.get('table_size'))}",
        f"table_quat [xyzw]={format_vector(meta.get('table_quat'), 4)}",
    ]


def build_subtitles(fps, command, drop, width, height, context=None):
    """Return ASS text and footer height; commands refresh at 0.3 s, xyz every frame."""
    context = context if context is not None else SceneContext()
    rate = Fraction(str(fps))
    duration = Fraction(len(command), 1) / rate
    fontsize = max(12, round(width / 1920 * 28))
    margin = max(8, round(width / 1920 * 24))
    # Explicitly wrap every block and reserve its maximum height, so even small
    # replay videos retain readable text without overlapping metadata or xyz.
    columns = max(1, int((width - 2 * margin) / (fontsize * 0.65)))

    def wrap(lines):
        return [
            wrapped for line in lines
            for wrapped in textwrap.wrap(line, width=columns, subsequent_indent="  ")
        ]

    static_lines = wrap(scene_lines(context))
    command_events = []
    for tick in range(math.ceil(duration / REFRESH)):
        sample_time = tick * REFRESH
        frame = min(int(sample_time * rate), len(command) - 1)
        dx, dyaw = command[frame]
        dx, dyaw = (0.0 if abs(v) < 0.0005 else float(v) for v in (dx, dyaw))
        lines = wrap([
            f"Sparse command | sample t={float(sample_time):.2f}s | frame={frame} | refresh=0.3s",
            f"dx={dx:+.3f} m    dyaw={dyaw:+.3f} rad    drop={float(drop[frame, 0]):.0f}",
        ])
        command_events.append((tick * 30, min((tick + 1) * 30, math.ceil(duration * 100)), lines))

    object_events = []
    for frame in range(len(command)):
        position = context.object_positions[frame] if context.object_positions is not None else None
        lines = wrap([
            f"Object current [world xyz,m] | t={frame / fps:.2f}s | frame={frame}: {format_vector(position)}"
        ])
        # Floor to ASS centiseconds so the event is already active at this
        # frame's presentation time, including non-centisecond rates like 30 Hz.
        start_cs = int(Fraction(frame * 100, 1) / rate)
        end_cs = (math.ceil(duration * 100) if frame == len(command) - 1
                  else int(Fraction((frame + 1) * 100, 1) / rate))
        object_events.append((start_cs, end_cs, lines))

    command_rows = max(len(lines) for _, _, lines in command_events)
    object_rows = max(len(lines) for _, _, lines in object_events)
    line_height = math.ceil(fontsize * 1.5)
    footer = 2 * margin + (command_rows + object_rows + len(static_lines)) * line_height
    footer += (height + footer) % 2
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height + footer}\nWrapStyle: 2\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,DejaVu Sans Mono,{fontsize},&H00FFFFFF,&H00FFFFFF,&H00000000,"
        f"&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,{margin},{margin},{margin},1\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []

    def event(start_cs, end_cs, lines, row):
        if end_cs <= start_cs:
            return
        position = rf"{{\an7\pos({margin},{height + margin + row * line_height})}}"
        text = r"\N".join(lines)
        events.append(
            f"Dialogue: 0,{ass_time(start_cs)},{ass_time(end_cs)},Default,,0,0,0,,{position}{text}\n"
        )

    event(0, math.ceil(duration * 100), static_lines, command_rows + object_rows)
    for start_cs, end_cs, lines in command_events:
        event(start_cs, end_cs, lines, 0)
    for start_cs, end_cs, lines in object_events:
        event(start_cs, end_cs, lines, command_rows)
    return header + "".join(events), footer


def annotate_video(source, output, commands, context=None):
    context = context if context is not None else SceneContext()
    fps, command, drop = commands
    stream = probe_video(source)
    validate_timeline(stream, fps, len(command))
    width, height = int(stream["width"]), int(stream["height"])
    subtitles, footer = build_subtitles(fps, command, drop, width, height, context)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination for atomic replacement after successful encoding.
    with tempfile.TemporaryDirectory(prefix=".sparse-vis-", dir=output.parent) as work:
        work = Path(work)
        (work / "commands.ass").write_text(subtitles, encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:v:0",
             "-map", "0:a?", "-vf", f"pad=iw:ih+{footer}:0:0:color=0x141820,ass=commands.ass",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "2",
             "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", "result.mp4"],
            check=True, cwd=work,
        )
        validate_timeline(probe_video(work / "result.mp4"), fps, len(command))
        (work / "result.mp4").replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--video-dir", type=Path, help="Source replay videos (default: DATASET/vis)")
    parser.add_argument("--output-dir", type=Path, help="Default: DATASET/vis_sparse_command")
    parser.add_argument("--motion", action="append", help="Select a filename stem; may be repeated")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all motions")
    parser.add_argument("--existing-only", action="store_true", help="Skip missing videos without launching Isaac Sim")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already annotated output files")
    parser.add_argument("--camera-offset", default="1.5,-1.5,1.0")
    parser.add_argument("--quat-convention", choices=("auto", "xyzw", "wxyz"), default="xyzw")
    parser.add_argument("--runtime", choices=("auto", "local", "server"), default="auto")
    parser.add_argument("--parallel-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.max_videos < 0 or args.parallel_jobs < 1:
        parser.error("max-videos must be nonnegative and parallel-jobs must be positive")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            parser.error(f"{tool} is required")
    dataset = args.dataset_dir.resolve()
    videos = (args.video_dir or dataset / "vis").resolve()
    output = (args.output_dir or dataset / "vis_sparse_command").resolve()
    if videos == output:
        parser.error("Source and output directories must differ to avoid stacking footers")
    paths = sorted((dataset / "robot").glob("*.pkl"))
    if args.motion:
        selected = set(args.motion)
        unknown = selected - {p.stem for p in paths}
        if unknown:
            parser.error(f"Unknown motions: {sorted(unknown)}")
        paths = [p for p in paths if p.stem in selected]
    if args.max_videos:
        paths = paths[:args.max_videos]
    if not paths:
        parser.error("No robot motions selected")
    done = skipped = unavailable = 0
    plan = []
    for p in paths:
        if args.skip_existing and (output / f"{p.stem}.mp4").is_file():
            skipped += 1
            continue
        plan.append((p, load_commands(p)))
    missing = [p for p, _ in plan if not (videos / f"{p.stem}.mp4").is_file()]
    if missing and not args.existing_only and videos != dataset / "vis":
        parser.error("Custom video-dir has missing videos; supply --existing-only or render them first")

    # Finish existing replays before launching Isaac Sim. Newly rendered replays
    # are annotated by each render worker immediately after its writer closes.
    for p, commands in plan:
        source, destination = videos / f"{p.stem}.mp4", output / f"{p.stem}.mp4"
        if not source.is_file():
            if args.existing_only:
                print(f"Missing replay video: {source}", flush=True)
                unavailable += 1
            continue
        print(f"Annotating {p.stem}", flush=True)
        context = load_scene_context(p, commands[0], len(commands[1]))
        annotate_video(source, destination, commands, context)
        done += 1
        print(f"Ready: {destination}", flush=True)

    if missing and not args.existing_only:
        # Use the same kinematic replay renderer and camera as place_data_generater/run.sh.
        # A temporary view limits rendering to the missing, selected motions.
        with tempfile.TemporaryDirectory(prefix="grail-sparse-render-") as work:
            view = Path(work)
            (view / "robot").mkdir()
            videos.mkdir(parents=True, exist_ok=True)
            for p in missing:
                (view / "robot" / p.name).symlink_to(p)
            for sub in ("objects", "object_usd", "meta"):
                (view / sub).symlink_to(dataset / sub, target_is_directory=True)
            (view / "vis").symlink_to(videos, target_is_directory=True)
            subprocess.run(
                ["bash", str(REPO_ROOT / "grail/visualization/scripts/visualize.sh"),
                 str(view), "0", args.camera_offset, args.quat_convention, args.runtime,
                 str(args.parallel_jobs), str(output)], check=True, cwd=REPO_ROOT,
            )
        for p in missing:
            destination = output / f"{p.stem}.mp4"
            if (videos / f"{p.stem}.mp4").is_file() and destination.is_file():
                done += 1
            else:
                print(f"Missing annotated video: {destination}", flush=True)
                unavailable += 1
    print(f"Done: {done} annotated, {skipped} skipped, {unavailable} missing. Output: {output}")
    if unavailable and not args.existing_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
