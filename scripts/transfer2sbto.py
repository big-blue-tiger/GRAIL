#!/usr/bin/env python3
"""Convert GRAIL motion data to SBTO input format.

This script converts motion capture data from GRAIL format (robot/*.pkl + objects/*.pkl)
into SBTO-compatible trajectory files. The conversion handles quaternion format differences,
removes hand data, and packs the data into SBTO's expected structure.

Usage:
    # Single file conversion
    python transfer2sbto.py \
        --robot robot/pickup_ground__alcohol_0__000.pkl \
        --object objects/pickup_ground__alcohol_0__000.pkl \
        --output output/alcohol_0.npz

    # Batch conversion with pattern
    python scripts/transfer2sbto.py --batch \
        --robot-dir data/hf_dataset/data/pickup_table/robot \
        --object-dir data/hf_dataset/data/pickup_table/objects \
        --output-dir data/grail2sbto_dataset_table 

    # Trajectory format (separate arrays like best_trajectory.npz)
    python transfer2sbto.py --batch \
        --robot-dir robot \
        --object-dir objects \
        --output-dir data/grail2sbto_dataset \
        --format trajectory
"""

import argparse
import fnmatch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    import joblib
except ImportError:
    raise ImportError("joblib is required. Install with: pip install joblib")


def load_grail_data(robot_pkl: Path, object_pkl: Path) -> Tuple[Dict, Dict, str, float]:
    """Load GRAIL robot and object data from PKL files.

    Args:
        robot_pkl: Path to robot PKL file
        object_pkl: Path to object PKL file

    Returns:
        Tuple of (robot_data, object_data, motion_name, fps)
    """
    if not robot_pkl.exists():
        raise FileNotFoundError(f"Robot file not found: {robot_pkl}")
    if not object_pkl.exists():
        raise FileNotFoundError(f"Object file not found: {object_pkl}")

    robot_dict = joblib.load(robot_pkl)
    object_dict = joblib.load(object_pkl)

    if not robot_dict:
        raise ValueError(f"Robot file is empty: {robot_pkl}")
    if not object_dict:
        raise ValueError(f"Object file is empty: {object_pkl}")

    robot_motion_name = list(robot_dict.keys())[0]
    object_motion_name = list(object_dict.keys())[0]

    if robot_motion_name != object_motion_name:
        print(f"Warning: Motion names don't match: robot={robot_motion_name}, object={object_motion_name}")

    robot_data = robot_dict[robot_motion_name]
    object_data = object_dict[object_motion_name]

    robot_fps = robot_data.get('fps', 25.0)
    object_fps = object_data.get('fps', 25.0)

    if abs(robot_fps - object_fps) > 0.01:
        raise ValueError(f"FPS mismatch: robot={robot_fps}, object={object_fps}")

    return robot_data, object_data, robot_motion_name, float(robot_fps)


def convert_to_qpos_format(robot_data: Dict, object_data: Dict) -> np.ndarray:
    """Convert GRAIL data to SBTO qpos format.

    SBTO qpos layout (T, 43):
        [0:4]   root quaternion (wxyz)
        [4:7]   root position (xyz)
        [7:36]  29 joint angles
        [36:40] object quaternion (wxyz)
        [40:43] object position (xyz)

    Args:
        robot_data: GRAIL robot motion data
        object_data: GRAIL object motion data

    Returns:
        qpos array with shape (T, 43)
    """
    root_pos = robot_data['root_trans_offset']  # (T, 3)
    root_rot_xyzw = robot_data['root_rot']       # (T, 4) XYZW format
    dof = robot_data['dof']                      # (T, 29)

    object_pos = object_data['root_pos']         # (T, 1, 3)
    object_quat_wxyz = object_data['root_quat']  # (T, 1, 4) WXYZ format

    num_frames = root_pos.shape[0]

    if root_rot_xyzw.shape[0] != num_frames or dof.shape[0] != num_frames:
        raise ValueError(f"Frame count mismatch: root_pos={num_frames}, root_rot={root_rot_xyzw.shape[0]}, dof={dof.shape[0]}")
    if object_pos.shape[0] != num_frames or object_quat_wxyz.shape[0] != num_frames:
        raise ValueError(f"Object frame count mismatch: expected {num_frames}, got pos={object_pos.shape[0]}, quat={object_quat_wxyz.shape[0]}")

    if dof.shape[1] != 29:
        raise ValueError(f"Expected 29 DOFs, got {dof.shape[1]}")

    qpos = np.zeros((num_frames, 43), dtype=np.float64)

    # Convert robot quaternion from XYZW to WXYZ
    qpos[:, 0:4] = root_rot_xyzw[:, [3, 0, 1, 2]]

    # Robot position
    qpos[:, 4:7] = root_pos

    # Joint angles (already in MuJoCo order)
    qpos[:, 7:36] = dof

    # Object quaternion (already WXYZ, remove singleton dimension)
    qpos[:, 36:40] = object_quat_wxyz[:, 0, :]

    # Object position (remove singleton dimension)
    qpos[:, 40:43] = object_pos[:, 0, :]

    return qpos


def convert_to_trajectory_format(robot_data: Dict, object_data: Dict) -> Dict[str, Any]:
    """Convert GRAIL data to SBTO trajectory format (separate arrays).

    This format matches SBTO's best_trajectory.npz output.

    Args:
        robot_data: GRAIL robot motion data
        object_data: GRAIL object motion data

    Returns:
        Dictionary with separate trajectory arrays
    """
    root_pos = robot_data['root_trans_offset']  # (T, 3)
    root_rot_xyzw = robot_data['root_rot']       # (T, 4) XYZW format
    dof_pos = robot_data['dof']                  # (T, 29)

    object_pos = object_data['root_pos'][:, 0, :]        # (T, 3)
    object_rot_wxyz = object_data['root_quat'][:, 0, :]  # (T, 4) WXYZ format

    num_frames = root_pos.shape[0]

    # Convert robot quaternion from XYZW to WXYZ
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

    return {
        'root_pos': root_pos.astype(np.float64),
        'root_rot': root_rot_wxyz.astype(np.float64),
        'dof_pos': dof_pos.astype(np.float64),
        'object_pos': object_pos.astype(np.float64),
        'object_rot': object_rot_wxyz.astype(np.float64),
        'time': np.arange(num_frames, dtype=np.float64),
    }


def convert_single_motion(
    robot_path: Path,
    object_path: Path,
    output_path: Path,
    output_format: str = 'qpos'
) -> bool:
    """Convert a single GRAIL motion to SBTO format.

    Args:
        robot_path: Path to robot PKL file
        object_path: Path to object PKL file
        output_path: Path to output NPZ file
        output_format: 'qpos' or 'trajectory'

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        robot_data, object_data, motion_name, fps = load_grail_data(robot_path, object_path)

        if output_format == 'qpos':
            qpos = convert_to_qpos_format(robot_data, object_data)
            output_data = {
                'qpos': qpos,
                'fps': np.array(fps, dtype=np.int64),
            }
            num_frames = qpos.shape[0]
        elif output_format == 'trajectory':
            trajectory_data = convert_to_trajectory_format(robot_data, object_data)
            output_data = trajectory_data
            output_data['fps'] = fps
            num_frames = trajectory_data['root_pos'].shape[0]
        else:
            raise ValueError(f"Unknown format: {output_format}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **output_data)

        print(f"✓ {robot_path.name} -> {output_path.name}")
        print(f"  Frames: {num_frames}, FPS: {fps}, Format: {output_format}")

        return True

    except Exception as e:
        print(f"✗ Failed: {robot_path.name}")
        print(f"  Error: {e}")
        return False


def discover_grail_files(
    robot_dir: Path,
    object_dir: Path,
    pattern: str = None
) -> List[Tuple[Path, Path, str]]:
    """Discover matching robot/object PKL file pairs.

    Args:
        robot_dir: Directory containing robot PKL files
        object_dir: Directory containing object PKL files
        pattern: Optional glob pattern for file selection (e.g., "*alcohol*")

    Returns:
        List of (robot_path, object_path, base_name) tuples
    """
    if not robot_dir.exists():
        raise FileNotFoundError(f"Robot directory not found: {robot_dir}")
    if not object_dir.exists():
        raise FileNotFoundError(f"Object directory not found: {object_dir}")

    robot_files = sorted(robot_dir.glob("*.pkl"))

    if pattern:
        robot_files = [f for f in robot_files if fnmatch.fnmatch(f.name, pattern)]

    pairs = []
    for robot_file in robot_files:
        base_name = robot_file.stem
        object_file = object_dir / f"{base_name}.pkl"

        if object_file.exists():
            pairs.append((robot_file, object_file, base_name))
        else:
            print(f"⚠ Skipped {robot_file.name}: no matching object file")

    return pairs


def batch_convert(
    robot_dir: Path,
    object_dir: Path,
    output_dir: Path,
    pattern: str = None,
    output_format: str = 'qpos'
) -> Dict[str, int]:
    """Batch convert GRAIL motions to SBTO format.

    Args:
        robot_dir: Directory containing robot PKL files
        object_dir: Directory containing object PKL files
        output_dir: Output directory for NPZ files
        pattern: Optional glob pattern for file selection
        output_format: 'qpos' or 'trajectory'

    Returns:
        Dictionary with conversion statistics
    """
    pairs = discover_grail_files(robot_dir, object_dir, pattern)

    if not pairs:
        print(f"No matching file pairs found in {robot_dir}")
        return {'success': 0, 'failed': 0, 'total': 0}

    print(f"Found {len(pairs)} motion pairs to convert")
    print(f"Output directory: {output_dir}")
    print(f"Format: {output_format}")
    print("-" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {'success': 0, 'failed': 0, 'total': len(pairs)}

    for robot_path, object_path, base_name in pairs:
        output_path = output_dir / f"{base_name}.npz"

        if convert_single_motion(robot_path, object_path, output_path, output_format):
            stats['success'] += 1
        else:
            stats['failed'] += 1

    print("-" * 70)
    print(f"Conversion complete:")
    print(f"  Success: {stats['success']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Total:   {stats['total']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert GRAIL motion data to SBTO format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file conversion
  python transfer2sbto.py \\
    --robot robot/pickup_ground__alcohol_0__000.pkl \\
    --object objects/pickup_ground__alcohol_0__000.pkl \\
    --output converted/alcohol_0.npz

  # Batch conversion with pattern
  python transfer2sbto.py --batch \\
    --robot-dir data/hf_dataset/data/pickup_ground/robot \\
    --object-dir data/hf_dataset/data/pickup_ground/objects \\
    --output-dir converted_sbto \\
    --pattern "*alcohol*"

  # Trajectory format output
  python transfer2sbto.py --batch \\
    --robot-dir robot \\
    --object-dir objects \\
    --output-dir converted \\
    --format trajectory
        """
    )

    parser.add_argument('--robot', type=Path, help='Input robot PKL file path')
    parser.add_argument('--object', type=Path, help='Input object PKL file path')
    parser.add_argument('--output', type=Path, help='Output NPZ file path')

    parser.add_argument('--batch', action='store_true', help='Enable batch processing mode')
    parser.add_argument('--robot-dir', type=Path, help='Batch: robot directory')
    parser.add_argument('--object-dir', type=Path, help='Batch: object directory')
    parser.add_argument('--output-dir', type=Path, help='Batch: output directory')
    parser.add_argument('--pattern', type=str, help='Batch: glob pattern for file selection (e.g., "*alcohol*")')

    parser.add_argument('--format', type=str, default='qpos', choices=['qpos', 'trajectory'],
                       help='Output format: qpos (packed array) or trajectory (separate arrays)')

    args = parser.parse_args()

    if args.batch:
        if not args.robot_dir or not args.object_dir or not args.output_dir:
            parser.error("--batch mode requires --robot-dir, --object-dir, and --output-dir")
        batch_convert(args.robot_dir, args.object_dir, args.output_dir, args.pattern, args.format)
    else:
        if not args.robot or not args.object or not args.output:
            parser.error("Single file mode requires --robot, --object, and --output")
        success = convert_single_motion(args.robot, args.object, args.output, args.format)
        exit(0 if success else 1)


if __name__ == '__main__':
    main()
