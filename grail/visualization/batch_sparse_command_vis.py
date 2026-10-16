#!/usr/bin/env python3
"""Add a sparse-command footer to replay videos, refreshing every 0.3 seconds."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import joblib
import numpy as np


REFRESH = Fraction(3, 10)
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def build_subtitles(fps, command, drop, width, height, footer_height):
    """Sample the source timeline at exact 0.3 s boundaries (no rounded frame stride)."""
    rate = Fraction(str(fps))
    duration = Fraction(len(command), 1) / rate
    ticks = math.ceil(duration / REFRESH)
    fontsize = max(12, round(width / 1920 * 30))
    margin = max(8, round(width / 1920 * 24))
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height + footer_height}\nWrapStyle: 2\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,DejaVu Sans Mono,{fontsize},&H00FFFFFF,&H00FFFFFF,&H00000000,"
        f"&H00000000,0,0,0,0,100,100,0,0,1,0,0,1,{margin},{margin},12,1\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for tick in range(ticks):
        sample_time = tick * REFRESH
        frame = min(int(sample_time * rate), len(command) - 1)
        dx, dyaw = command[frame]
        # Three decimals are readable without showing a negative zero.
        dx, dyaw = (0.0 if abs(v) < 0.0005 else float(v) for v in (dx, dyaw))
        text = (
            f"Sparse command | sample t={float(sample_time):.2f}s | frame={frame} | refresh=0.3s"
            r"\N"
            f"dx={dx:+.3f} m    dyaw={dyaw:+.3f} rad    drop={float(drop[frame, 0]):.0f}"
        )
        # ASS timestamps use centiseconds. Keep the last event through the final frame.
        end_cs = min((tick + 1) * 30, math.ceil(duration * 100))
        events.append(f"Dialogue: 0,{ass_time(tick * 30)},{ass_time(end_cs)},Default,,0,0,0,,{text}\n")
    return header + "".join(events)


def annotate_video(source, output, commands):
    fps, command, drop = commands
    stream = probe_video(source)
    validate_timeline(stream, fps, len(command))
    width, height = int(stream["width"]), int(stream["height"])
    footer = max(64, round(width / 1920 * 112))
    footer += (height + footer) % 2
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination for atomic replacement after successful encoding.
    with tempfile.TemporaryDirectory(prefix=".sparse-vis-", dir=output.parent) as work:
        work = Path(work)
        (work / "commands.ass").write_text(
            build_subtitles(fps, command, drop, width, height, footer)
        )
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
    plan = [(p, load_commands(p)) for p in paths]
    missing = [p for p, _ in plan if not (videos / f"{p.stem}.mp4").is_file()]
    if missing and not args.existing_only:
        if videos != dataset / "vis":
            parser.error("Custom video-dir has missing videos; supply --existing-only or render them first")
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
                 str(args.parallel_jobs)], check=True, cwd=REPO_ROOT,
            )
    done = skipped = unavailable = 0
    for p, commands in plan:
        source, destination = videos / f"{p.stem}.mp4", output / f"{p.stem}.mp4"
        if not source.is_file():
            print(f"Missing replay video: {source}", flush=True)
            unavailable += 1
            continue
        if args.skip_existing and destination.is_file():
            skipped += 1
            continue
        print(f"Annotating {p.stem}", flush=True)
        annotate_video(source, destination, commands)
        done += 1
    print(f"Done: {done} annotated, {skipped} skipped, {unavailable} missing. Output: {output}")
    if unavailable and not args.existing_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
