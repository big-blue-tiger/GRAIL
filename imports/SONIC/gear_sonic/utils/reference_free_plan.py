"""Small, reference-independent carry plans calibrated on teacher rollouts.

Only NumPy is required. The virtual root path is discarded after extracting
25 Hz sparse commands and the two fixed object goals.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CarryPlannerConfig:
    lift_range: tuple = (0.20, 0.35)
    lower_range: tuple = (0.05, 0.14)
    distance_range: tuple = (2.5, 3.2)
    sector_degrees: float = 75.0
    angular_speed: float = 1.0
    turn_start_speed: float = -0.15  # Signed speed at the start of the turn.
    turn_forward_speed: float = 0.4  # Ramp endpoint; also matches walk_speed.
    walk_speed: float = 0.4
    pickup_settle_s: float = 1.0
    hold_s: float = 1.2

    def __post_init__(self):
        for name in ("lift_range", "lower_range", "distance_range"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all() or not 0 < values[0] <= values[1]:
                raise ValueError(f"{name} must contain two finite positive increasing values")
        for name in ("angular_speed", "walk_speed", "hold_s"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("turn_forward_speed", "pickup_settle_s"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.turn_start_speed):
            raise ValueError("turn_start_speed must be finite")
        if not np.isfinite(self.sector_degrees) or not 0 < self.sector_degrees < 90:
            raise ValueError("sector_degrees must be between 0 and 90")


PLAN_FPS = 25
CARRY_OFFSET = np.array([0.39, -0.09])
RELEASE_OFFSET = np.array([0.33, -0.11])


@dataclass(frozen=True)
class CarryPlan:
    carry_position: np.ndarray
    carry_quaternion: np.ndarray
    release_position: np.ndarray
    release_quaternion: np.ndarray
    sparse: np.ndarray
    drop: np.ndarray
    turn_end_frame: int
    walk_end_frame: int
    release_start_frame: int
    turn_angle: float
    straight_length: float


def rotation_2d(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def table_outward_normal(table_quaternion, table_size, robot_offset):
    """Match the generator's wxyz convention and choose the robot's table side."""
    q = np.asarray(table_quaternion, dtype=float)
    size = np.asarray(table_size, dtype=float)
    if (q.shape != (4,) or size.shape != (3,) or not np.isfinite(q).all()
            or not np.isfinite(size).all() or np.any(size <= 0)
            or abs(np.linalg.norm(q) - 1) > 0.01):
        raise ValueError("Invalid table quaternion or size")
    w, x, y, z = q / np.linalg.norm(q)
    rot = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    if abs(rot[2, 2]) < 0.999:
        raise ValueError("Carry planning requires a horizontal table")
    normal = rot[:2, int(np.argmin(size[:2]))]
    side = np.dot(normal, np.asarray(robot_offset)[:2])
    return normal if side >= 0 else -normal


def _root_path(initial_yaw, angle, length, config):
    """Return planar positions/headings through hold and the first release frame."""
    nturn = max(1, round(abs(angle) / config.angular_speed * PLAN_FPS)) if angle else 0
    nwalk = max(1, round(length / config.walk_speed * PLAN_FPS))
    walk_end = nturn + nwalk
    release = walk_end + max(1, round(config.hold_s * PLAN_FPS))
    times = np.arange(release + 1) / PLAN_FPS
    yaw = np.full(len(times), initial_yaw + angle)
    positions = np.zeros((len(times), 2))
    if nturn:
        omega = angle / (nturn / PLAN_FPS)
        turning_yaw = initial_yaw + omega * np.minimum(times, nturn / PLAN_FPS)
        # Rollouts step back first, then accelerate forward during the turn.
        progress = (np.arange(nturn) + 0.5) / nturn
        speed = config.turn_start_speed + (config.turn_forward_speed - config.turn_start_speed) * progress
        heading = initial_yaw + angle * progress
        velocity = speed[:, None] * np.column_stack((np.cos(heading), np.sin(heading)))
        positions[1:nturn + 1] = np.cumsum(velocity / PLAN_FPS, axis=0)
        positions[nturn + 1:] = positions[nturn]
        yaw[:nturn] = turning_yaw[:nturn]
    progress = np.clip((np.arange(len(times)) - nturn) / nwalk, 0, 1)
    positions += progress[:, None] * length * rotation_2d(initial_yaw + angle)[:, 0]
    return positions, yaw, nturn, walk_end, release


def plan_carry_task(object_position, object_quaternion, initial_yaw, final_yaw,
                    lift, lower, distance, config=CarryPlannerConfig()):
    """Build a deterministic arc/straight plan; goals exclude reset height lifts."""
    pos = np.asarray(object_position, dtype=float)
    quat = np.asarray(object_quaternion, dtype=float)
    if (pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all()
            or not np.isfinite(quat).all() or abs(np.linalg.norm(quat) - 1) > 0.01
            or not np.isfinite([initial_yaw, final_yaw, lift, lower, distance]).all()
            or lift <= 0 or lower < 0 or distance <= 0):
        raise ValueError("Invalid carry planning inputs")
    quat = quat / np.linalg.norm(quat)
    # Use the shortest signed turn: positive yaw is left, negative yaw is right,
    # matching the unwrapped heading convention in concat.turn_stage_commands.
    angle = float(np.arctan2(np.sin(final_yaw-initial_yaw), np.cos(final_yaw-initial_yaw)))
    if abs(angle) <= 0.2:
        angle = 0.0
    final_yaw = initial_yaw + angle
    # Compute the quantized arc first; its endpoint is independent of walk length.
    arc, _, nturn, _, _ = _root_path(initial_yaw, angle, 1.0, config)
    start_rot, end_rot = rotation_2d(initial_yaw), rotation_2d(final_yaw)
    direction = end_rot[:, 0]
    correction = arc[nturn] + end_rot @ RELEASE_OFFSET - start_rot @ CARRY_OFFSET
    forward = float(correction @ direction)
    perpendicular = correction - forward * direction
    radicand = distance**2 - float(perpendicular @ perpendicular)
    if radicand <= 0:
        raise ValueError("Transport distance is too short for the planned turn")
    length = float(np.sqrt(radicand) - forward)
    if length <= 0:
        raise ValueError("Planned straight length must be positive")
    path, yaw, nturn, walk_end, release = _root_path(initial_yaw, angle, length, config)

    carry_pos = pos.copy()
    carry_pos[2] += lift
    release_pos = carry_pos.copy()
    release_pos[:2] += path[-1] + end_rot @ RELEASE_OFFSET - start_rot @ CARRY_OFFSET
    release_pos[2] -= lower
    # World-yaw rotation times the original wxyz object orientation.
    c, s = np.cos(angle/2), np.sin(angle/2)
    w, x, y, z = quat
    release_quat = np.array([c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w])

    # Same projection, lookahead, XY deadzone and length clamp as concat.py.
    future = np.minimum(np.arange(len(path)) + round(0.5 * PLAN_FPS), release-1)
    delta = path[future] - path
    c, s = np.cos(yaw), np.sin(yaw)
    xy = np.column_stack((c*delta[:, 0]+s*delta[:, 1], -s*delta[:, 0]+c*delta[:, 1]))
    norms = np.linalg.norm(xy, axis=1)
    xy[norms < 0.02] = 0
    xy *= np.minimum(1, 0.15 / np.maximum(norms, 1e-12))[:, None]
    sparse = np.zeros((len(path), 2), dtype=np.float32)
    sparse[:release, 0] = xy[:release, 0]
    sparse[:nturn, 1] = angle
    drop = np.zeros((len(path), 1), dtype=np.float32)
    drop[release:] = 1
    return CarryPlan(
        carry_pos.astype(np.float32), quat.astype(np.float32),
        release_pos.astype(np.float32), release_quat.astype(np.float32),
        sparse, drop, nturn, walk_end, release, angle, length,
    )


def sample_carry_plan(object_position, object_quaternion, initial_yaw, outward_normal,
                      rng, config=CarryPlannerConfig()):
    """Sample an outward goal and take the shortest turn to its heading."""
    normal_yaw = np.arctan2(outward_normal[1], outward_normal[0])
    sector = np.deg2rad(config.sector_degrees)
    for _ in range(128):
        final_yaw = normal_yaw + rng.uniform(-sector, sector)
        plan = plan_carry_task(
            object_position, object_quaternion, initial_yaw, final_yaw,
            rng.uniform(*config.lift_range), rng.uniform(*config.lower_range),
            np.sqrt(rng.uniform(*(np.square(config.distance_range)))), config,
        )
        displacement = plan.release_position[:2] - plan.carry_position[:2]
        bearing = np.arctan2(displacement[1], displacement[0]) - normal_yaw
        if abs(np.arctan2(np.sin(bearing), np.cos(bearing))) <= sector:
            return plan
    raise ValueError("Cannot sample a release goal in the outward table sector")
