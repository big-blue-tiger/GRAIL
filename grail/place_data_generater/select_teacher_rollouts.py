#!/usr/bin/env python3
"""按优先级选取 teacher rollout，并复制完整配套数据。

默认选 1800 条（优先左/右各 900 条），路径相对于项目根目录，可从任意目录启动：

    python grail/place_data_generater/select_teacher_rollouts.py --dry-run
    python grail/place_data_generater/select_teacher_rollouts.py

先排除不满足以下硬性要求的数据：物体在 drop 首次为 1 前，z < 0.2 m
连续达到 0.1 秒即视为提前落地；carry goal 与物体第 0 帧位置的三维距离必须 > 0.07 m。
地面按数据世界坐标 z=0 计算，连续帧数为 ceil(0.1 * fps)，drop 触发帧不参与检测。
新检查直接使用 objects.root_pos、robot.policy_command_drop_button 和
meta.place_object_carry_goal_pos_w，不推断缺失字段。

物体到 release goal 的最小三维距离读取 rollout.min_release_goal_distance_m，
在左转和右转内部各自按该值升序选取，不设距离门槛；缺失或无效时排在最后。
方向采用实际 rollout 的 turn_direction，角度采用 abs(yaw_change_rad)，
碰撞采用 robot_table_collision；缺少转向统计时由机器人 xyzw 四元数累计 yaw。
距离为 robot.root_trans_offset 在 grail_start_frame 和 output_contact_start_frame
两帧的 XY 欧氏距离；优先读取 meta 中已裁剪的帧号，其次读取 reports。

仅在通过硬性要求的数据中固定左右配额，每个方向严格依次比较：release goal 最小距离升序、
角度 < 180°、明确无碰撞、
不属于全体有效距离中最小的 5%（向上取整）、距离从小到大、文件名。
因此只有 release goal 最小距离相同时，才进一步比较角度、碰撞及 base XY 距离。
后面条件不会覆盖前面条件；最小 5% 仍保留为候补，必要时也会复制。
某个方向不足时，用另一方向补足，最后才用直行/未知方向；总数不足则报错。
输出目录必须不存在。仅复制，保留源文件，包括 USD 纹理及共享 BPS basis。
依赖：numpy、joblib，不需要启动仿真器。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path("data/hf_dataset/data_update/data")
FOLDERS = ("objects", "object_usd", "meta", "bps", "robot", "reports")


@dataclass
class Candidate:
    name: str
    direction: str
    yaw_change_rad: float | None
    robot_table_collision: bool | None
    base_xy_distance_m: float | None
    grail_start_frame: int | None
    output_contact_start_frame: int | None
    rollout: dict
    carry_init_distance_m: float
    first_drop_frame: int
    early_grounded_frame: int | None
    min_release_goal_distance_m: float | None = None
    notes: list[str] = field(default_factory=list)
    bottom_five_percent: bool = False

    @property
    def object_requirements_ok(self):
        return self.early_grounded_frame is None and self.carry_init_distance_m > 0.07

    @property
    def angle_ok(self):
        return self.yaw_change_rad is not None and abs(self.yaw_change_rad) < math.pi

    @property
    def rank(self):
        # 第 5 优先级：正常距离、最小 5% 候补、未知距离。
        distance_group = (2 if self.base_xy_distance_m is None
                          else 1 if self.bottom_five_percent else 0)
        return (
            self.min_release_goal_distance_m if self.min_release_goal_distance_m is not None else math.inf,
            not self.angle_ok,
            self.robot_table_collision is not False,
            distance_group,
            self.base_xy_distance_m if self.base_xy_distance_m is not None else math.inf,
            self.name,
        )

    def summary(self):
        return {
            "name": self.name,
            "carry_init_distance_m": self.carry_init_distance_m,
            "first_drop_frame": self.first_drop_frame,
            "early_grounded_frame": self.early_grounded_frame,
            "object_requirements_ok": self.object_requirements_ok,
            "min_release_goal_distance_m": self.min_release_goal_distance_m,
            "direction": self.direction,
            "yaw_change_rad": self.yaw_change_rad,
            "turn_angle_deg": (math.degrees(abs(self.yaw_change_rad))
                               if self.yaw_change_rad is not None else None),
            "angle_under_180": self.angle_ok,
            "robot_table_collision": self.robot_table_collision,
            "grail_start_frame": self.grail_start_frame,
            "output_contact_start_frame": self.output_contact_start_frame,
            "base_xy_distance_m": self.base_xy_distance_m,
            "bottom_five_percent": self.bottom_five_percent,
            "notes": self.notes,
        }


def finite_number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def frame_number(meta, report, *keys):
    for mapping in (meta, report):
        for key in keys:
            value = finite_number(mapping.get(key))
            if value is not None and value >= 0 and value.is_integer():
                return int(value)
    return None


def object_safety_metrics(robot, objects, meta):
    positions = np.asarray(objects["root_pos"], dtype=float)
    drop = np.asarray(robot["policy_command_drop_button"], dtype=float)
    carry_goal = np.asarray(meta["place_object_carry_goal_pos_w"], dtype=float)
    fps = float(robot["fps"])
    total = len(robot["root_trans_offset"])
    if (positions.shape != (total, 1, 3) or drop.shape != (total, 1)
            or carry_goal.shape != (3,) or not np.isfinite(positions).all()
            or not np.isfinite(carry_goal).all() or not np.isin(drop, (0, 1)).all()
            or not math.isfinite(fps) or fps <= 0 or float(objects["fps"]) != fps):
        raise ValueError("物体筛选要求同帧数/FPS的有效轨迹、[T,1]二值drop和三维carry goal")
    drop_frames = np.flatnonzero(drop[:, 0] == 1)
    if not len(drop_frames):
        raise ValueError("物体筛选要求轨迹中存在 drop=1 的释放帧")
    first_drop = int(drop_frames[0])
    carry_distance = float(np.linalg.norm(carry_goal - positions[0, 0]))
    ground_frames = math.ceil(0.1 * fps)
    consecutive = 0
    for frame, z in enumerate(positions[:first_drop, 0, 2]):
        consecutive = consecutive + 1 if z < 0.2 else 0
        if consecutive >= ground_frames:
            return carry_distance, first_drop, frame - ground_frames + 1
    return carry_distance, first_drop, None


def read_candidate(source, name, aggregate):
    notes = []
    try:
        report = json.loads((source / "reports" / f"{name}.json").read_text())
        if not isinstance(report, dict):
            raise ValueError("report 必须为字典")
    except (OSError, ValueError) as error:
        report = {}
        notes.append(f"无法读取 report: {error}")
    # 单条报告覆盖汇总报告：汇总可能只包含最近一次导出的分片。
    rollout = dict(aggregate.get(name, {}))
    rollout.update(report.get("rollout", {}))
    goal_distance = finite_number(rollout.get("min_release_goal_distance_m"))
    if goal_distance is None or goal_distance < 0:
        goal_distance = None
        notes.append("缺少可靠的物体到 release goal 最小距离，在同方向内排在有效距离之后")
    yaw_change = finite_number(rollout.get("yaw_change_rad"))
    collision = rollout.get("robot_table_collision")
    if not isinstance(collision, bool):
        collision = None
        notes.append("缺少可靠的机器人—桌面碰撞统计，按未满足无碰撞条件处理")

    try:
        meta = joblib.load(source / "meta" / f"{name}.pkl")
        if not isinstance(meta, dict):
            raise ValueError("meta 必须为字典")
    except Exception as error:
        meta = {}
        notes.append(f"无法读取 meta: {error}")
    start = frame_number(meta, report, "grail_start_frame")
    contact = frame_number(meta, report, "place_output_contact_start_frame",
                           "output_contact_start_frame")
    distance = None
    try:
        payload = joblib.load(source / "robot" / f"{name}.pkl")
        if not isinstance(payload, dict) or len(payload) != 1:
            raise ValueError("robot PKL 应包含唯一的 motion")
        robot = next(iter(payload.values()))
    except Exception as error:
        robot = {}
        notes.append(f"无法读取 robot: {error}")
    object_payload = joblib.load(source / "objects" / f"{name}.pkl")
    (objects,) = object_payload.values()
    carry_distance, first_drop, early_grounded = object_safety_metrics(robot, objects, meta)
    if yaw_change is None:
        try:
            quats = np.asarray(robot["root_rot"], dtype=float)
            if quats.ndim != 2 or quats.shape[1] != 4 or not len(quats):
                raise ValueError("root_rot 应为非空的 [T,4] xyzw 四元数")
            norms = np.linalg.norm(quats, axis=1)
            if not np.isfinite(quats).all() or np.any(norms < 1e-8):
                raise ValueError("root_rot 包含无效四元数")
            x, y, z, w = (quats / norms[:, None]).T
            yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            yaw_change = float(yaw[-1] - yaw[0])
        except (KeyError, TypeError, ValueError) as error:
            notes.append(f"无法从四元数计算转角: {error}")
    try:
        positions = np.asarray(robot["root_trans_offset"], dtype=float)
        if positions.ndim != 2 or positions.shape[1] < 2:
            raise ValueError("root_trans_offset 应为 [T,3] base 坐标")
        if start is None or contact is None or max(start, contact) >= len(positions):
            raise ValueError(f"帧号缺失或越界: grail={start}, contact={contact}, T={len(positions)}")
        distance = finite_number(np.linalg.norm(positions[start, :2] - positions[contact, :2]))
        if distance is None:
            raise ValueError("两帧间的 base XY 距离不是有限数值")
    except (KeyError, TypeError, ValueError) as error:
        notes.append(f"无法计算 base XY 距离: {error}")

    direction = rollout.get("turn_direction")
    if direction not in {"left", "right", "straight"}:
        direction = ("unknown" if yaw_change is None else "left" if yaw_change > 0.02
                     else "right" if yaw_change < -0.02 else "straight")
    return Candidate(name, direction, yaw_change, collision, distance, start, contact,
                     rollout, carry_distance, first_drop, early_grounded,
                     min_release_goal_distance_m=goal_distance, notes=notes)


def collect_candidates(source):
    for folder in FOLDERS:
        if not (source / folder).is_dir():
            raise ValueError(f"缺少目录: {source / folder}")
    aggregate_path = source / "rollout_report.json"
    aggregate = json.loads(aggregate_path.read_text()) if aggregate_path.is_file() else {}
    candidates, skipped = [], []
    for path in sorted((source / "robot").glob("*.pkl")):
        name = path.stem
        required = [source / folder / f"{name}{suffix}" for folder, suffix in (
            ("objects", ".pkl"), ("meta", ".pkl"), ("bps", ".npy"), ("reports", ".json"))]
        missing = [str(p.relative_to(source)) for p in required if not p.is_file()]
        if not any((source / "object_usd" / f"{name}{suffix}").is_file()
                   for suffix in (".usd", ".usda", ".usdc")):
            missing.append(f"object_usd/{name}.usd[ac]")
        if missing:
            skipped.append({"name": name, "missing": missing})
            continue
        candidates.append(read_candidate(source, name, aggregate))
    return candidates, skipped


def select_candidates(candidates, count):
    candidates = [c for c in candidates if c.object_requirements_ok]
    if len(candidates) < count:
        raise ValueError(f"通过物体落地和 carry-init 距离检查的数据只有 {len(candidates)} 条，"
                         f"无法选出 {count} 条；未创建输出目录")
    distances = sorted((c for c in candidates if c.base_xy_distance_m is not None),
                       key=lambda c: (c.base_xy_distance_m, c.name))
    bottom_count = math.ceil(len(distances) * 0.05)
    for candidate in candidates:
        candidate.bottom_five_percent = False
    for candidate in distances[:bottom_count]:
        candidate.bottom_five_percent = True

    selected = []
    for direction in ("left", "right"):
        pool = sorted((c for c in candidates if c.direction == direction), key=lambda c: c.rank)
        selected.extend(pool[:count // 2])
    chosen = {c.name for c in selected}
    # 配额不足时先用另一转向补足，仍不足时才使用 straight / unknown。
    remaining = sorted((c for c in candidates if c.name not in chosen),
                       key=lambda c: (c.direction not in {"left", "right"}, c.rank))
    selected.extend(remaining[:count - len(selected)])
    return selected, bottom_count


def paired_assets(source, selected):
    names = {c.name for c in selected}
    assets = []
    for folder in FOLDERS:
        for path in sorted((source / folder).iterdir()):
            if path.name in names or path.stem in names:
                assets.append(path)
    for name in sorted(names):
        texture = source / "object_usd" / "textures" / name
        if texture.is_dir():
            assets.append(texture)
    for relative in ("bps/_basis.npy", "object_usd/config.yaml"):
        path = source / relative
        if path.is_file():
            assets.append(path)
    return assets


def run(source, output, count, dry_run=False):
    if count <= 0 or count % 2:
        raise ValueError("--count 必须为正偶数，以便设置相等的左右配额")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输入和输出目录必须相互独立")
    if output.exists():
        raise ValueError(f"输出目录已存在，请指定一个新目录: {output}")
    candidates, skipped = collect_candidates(source)
    rejected = [c for c in candidates if not c.object_requirements_ok]
    available = dict(Counter(c.direction for c in candidates))
    print(f"配套完整: {len(candidates)} 条，方向分布: {available}，缺配套跳过: {len(skipped)} 条")
    early_grounded_count = sum(c.early_grounded_frame is not None for c in candidates)
    carry_too_close_count = sum(c.carry_init_distance_m <= 0.07 for c in candidates)
    print(f"物体硬性筛选通过: {len(candidates) - len(rejected)} 条，排除: {len(rejected)} 条"
          f"（提前落地: {early_grounded_count}，carry-init 距离 <= 0.07 m: {carry_too_close_count}，可重叠）")
    selected, bottom_count = select_candidates(candidates, count)
    selected_counts = dict(Counter(c.direction for c in selected))
    summary = {
        "requested_count": count,
        "requested_per_direction": count // 2,
        "available_count": len(candidates),
        "available_directions": available,
        "object_eligible_count": len(candidates) - len(rejected),
        "object_rejected_count": len(rejected),
        "early_grounded_count": early_grounded_count,
        "carry_init_too_close_count": carry_too_close_count,
        "selected_count": len(selected),
        "selected_directions": selected_counts,
        "release_goal_distance_unknown_count": sum(c.min_release_goal_distance_m is None for c in selected),
        "angle_under_180_count": sum(c.angle_ok for c in selected),
        "collision_free_count": sum(c.robot_table_collision is False for c in selected),
        "collision_count": sum(c.robot_table_collision is True for c in selected),
        "collision_unknown_count": sum(c.robot_table_collision is None for c in selected),
        "bottom_five_percent_pool_count": bottom_count,
        "bottom_five_percent_selected_count": sum(c.bottom_five_percent for c in selected),
        "distance_unknown_count": sum(c.base_xy_distance_m is None for c in selected),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if any(selected_counts.get(direction, 0) != count // 2 for direction in ("left", "right")):
        print("警告：某个转向的数据不足，左右配额未满足，已使用其他数据补足总数。",
              file=sys.stderr)
    assets = paired_assets(source, selected)
    if dry_run:
        print(f"预览完成，将复制 {len(assets)} 个文件/目录到 {output}；未写入任何文件。")
        return summary

    output.mkdir(parents=True, exist_ok=False)
    for index, path in enumerate(assets, 1):
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
        if index % 300 == 0 or index == len(assets):
            print(f"已复制 {index}/{len(assets)} 个文件/目录", flush=True)
    manifest = {
        "source": os.path.relpath(source, output),
        "selection_policy": ["硬性排除 drop 首次为 1 前 z < 0.2 m 连续 ceil(0.1 * fps) 帧的数据",
                             "硬性要求 norm(carry_goal_xyz - object_init_xyz) > 0.07 m",
                             "通过硬性要求后左右各半，不足时仅用合格数据补足总数",
                             "各方向内 min_release_goal_distance_m 升序，未知距离排最后",
                             "abs(yaw_change_rad) < pi",
                             "robot_table_collision is false",
                             "全局距离最小 5% 留作候补，其余距离升序"],
        "distance_definition": "norm(base_xy[grail_start_frame] - base_xy[output_contact_start_frame])",
        "summary": summary,
        "selected": [c.summary() for c in selected],
        "rejected_object_requirements": [c.summary() for c in rejected],
        "skipped_incomplete": skipped,
    }
    (output / "selection_report.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "rollout_report.json").write_text(
        json.dumps({c.name: c.rollout for c in selected}, indent=2, ensure_ascii=False) + "\n")
    print(f"完成：已复制 {len(selected)} 条数据到 {output}")
    return summary


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=project_path,
                        default=DATA_DIR / "pickup_table_place_concat_slow_turn_teacher_rollout",
                        help="输入目录，相对路径按项目根目录解析")
    parser.add_argument("--output-dir", type=project_path,
                        default=DATA_DIR / "pickup_table_slowturn_1600",
                        help="新的输出目录，相对路径按项目根目录解析")
    parser.add_argument("--count", type=int, default=1600, help="总数，默认 1800，必须为正偶数")
    parser.add_argument("--dry-run", action="store_true", help="只展示筛选统计，不复制或创建目录")
    args = parser.parse_args()
    try:
        run(project_path(args.source_dir), project_path(args.output_dir), args.count, args.dry_run)
    except (OSError, ValueError) as error:
        parser.exit(1, f"错误：{error}\n")


if __name__ == "__main__":
    main()
