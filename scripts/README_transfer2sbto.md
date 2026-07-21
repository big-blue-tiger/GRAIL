# GRAIL to SBTO Data Conversion

This script converts GRAIL motion data to SBTO input format.

## Quick Start

### Single File Conversion

```bash
python3 scripts/transfer2sbto.py \
  --robot data/hf_dataset/data/pickup_ground/robot/pickup_ground__alcohol_0__000.pkl \
  --object data/hf_dataset/data/pickup_ground/objects/pickup_ground__alcohol_0__000.pkl \
  --output converted/alcohol_0.npz
```

### Batch Conversion

```bash
# Convert all alcohol files
python3 scripts/transfer2sbto.py --batch \
  --robot-dir data/hf_dataset/data/pickup_ground/robot \
  --object-dir data/hf_dataset/data/pickup_ground/objects \
  --output-dir converted_sbto \
  --pattern "*alcohol*"

# Convert specific numbered sequences
python3 scripts/transfer2sbto.py --batch \
  --robot-dir data/hf_dataset/data/pickup_ground/robot \
  --object-dir data/hf_dataset/data/pickup_ground/objects \
  --output-dir converted_sbto \
  --pattern "*alcohol_1[0-5]__*"
```

### Output Format Options

**QPOS format** (default) - Packed array matching SBTO input:
```bash
python3 scripts/transfer2sbto.py --batch \
  --robot-dir robot \
  --object-dir objects \
  --output-dir converted \
  --format qpos
```

Output: `qpos` array (T, 43)
- [0:4] root quaternion (wxyz)
- [4:7] root position (xyz)
- [7:36] 29 joint angles
- [36:40] object quaternion (wxyz)
- [40:43] object position (xyz)

**Trajectory format** - Separate arrays like best_trajectory.npz:
```bash
python3 scripts/transfer2sbto.py --batch \
  --robot-dir robot \
  --object-dir objects \
  --output-dir converted \
  --format trajectory
```

Output fields:
- `root_pos` (T, 3)
- `root_rot` (T, 4) - wxyz
- `dof_pos` (T, 29)
- `object_pos` (T, 3)
- `object_rot` (T, 4) - wxyz
- `time` (T,)
- `fps` (scalar)

## Data Format Details

### GRAIL Format (Source)
- Robot quaternion: **XYZW** format
- Object quaternion: **WXYZ** format
- Object data has singleton dimension: (T, 1, 3), (T, 1, 4)
- Includes hand data (14 DOFs) - ignored in conversion

### SBTO Format (Target)
- All quaternions: **WXYZ** format (MuJoCo standard)
- No singleton dimensions
- Body joints only (29 DOFs, no hand data)

### Conversion Details
- Robot quaternion: XYZW → WXYZ reordering
- Object quaternion: Direct copy (already WXYZ)
- Object position/quaternion: Remove singleton dimension
- Hand data: Completely excluded
- Joint angles: Direct copy (same MuJoCo order)

## Verification

The script has been tested and verified:
- ✅ Quaternion conversion XYZW→WXYZ correct
- ✅ Object data singleton dimension removal correct
- ✅ All quaternions normalized (norm=1.0)
- ✅ Batch processing with pattern matching works
- ✅ Both output formats produce valid data

### Example Verification

```python
import numpy as np

# Load converted data
data = np.load('converted/alcohol_0.npz')
qpos = data['qpos']

print(f"Shape: {qpos.shape}")  # (379, 43)
print(f"Root quat (WXYZ): {qpos[0, 0:4]}")
print(f"Root pos: {qpos[0, 4:7]}")
print(f"Object quat (WXYZ): {qpos[0, 36:40]}")
print(f"Object pos: {qpos[0, 40:43]}")
```

## Usage in SBTO

The converted NPZ files can be used directly as SBTO input:

```bash
# Use with SBTO visualization
python3 scripts/visualize_ref.py \
  --input converted/alcohol_0.npz \
  --model mj_model.xml
```

## Pattern Examples

- `"*"` - Convert all files
- `"*alcohol*"` - All files containing "alcohol"
- `"*_0__*"` - All files with "_0__" in name
- `"*alcohol_1[0-5]__*"` - Alcohol files numbered 10-15
- `"pickup_ground__*"` - All files starting with "pickup_ground__"
