# Retargeting

Convert GRAIL 4D HOI reconstructions (the output of `grail.pipelines.recon_4dhoi`) into
G1 robot motion trajectories consumable by the task-general tracking stack
under {src}`imports/SONIC/`.

The retargeting pipeline is a standalone subpackage:
{src}`grail/retargeting/`. All steps run as plain CLI tools.

## Install

1. Initialize the two submodules:
   ```bash
   git submodule update --init imports/GMR
   ```

2. Create a conda env with IsaacLab + IsaacSim + PyTorch (CUDA) following
   {blob}`imports/SONIC/README.md`. The default env
   name is `sonic`.

3. Install the retargeting stack on top:
   ```bash
   bash scripts/setup/install_env_sonic.sh
   ```

   This applies GRAIL-specific {blob}`GMR overrides <grail/retargeting/gmr_overrides/README.md>`
   on top of the public [YanjieZe/GMR](https://github.com/YanjieZe/GMR)
   submodule, then pip-installs GMR, GRAIL, and retargeting Python deps
   (`smplx`, `mujoco`, `usd-core`, …) into the active conda env. The script is
   idempotent — rerun it any time the submodule pin is bumped.

   Override the env name via `GRAIL_SONIC_ENV=<name>`:
   ```bash
   GRAIL_SONIC_ENV=my_sonic_env bash scripts/setup/install_env_sonic.sh
   ```

## Running the pipeline

### End-to-end (recommended)

```bash
conda activate sonic
export DISPLAY=:1                    # GMR uses mujoco viewer, needs a display

bash grail/retargeting/scripts/retarget_pipeline.sh \
    data/genhoi/benchmark_v3/generation/4dhoi_recon_valid/Hunyuan \
    benchmark_v3_0203
```

Outputs under `data/motion_lib/benchmark_v3_0203/`:

| Directory  | Contents                                          |
|------------|---------------------------------------------------|
| `robot/`   | G1 joint trajectories (one pkl per motion)        |
| `objects/` | Object 6-DOF trajectories                         |
| `object_usd/` | IsaacLab-ready USD assets                      |
| `meta/`    | Scene metadata (table pose, object name, …)       |

Plus a preprocessed twin at `data/motion_lib/benchmark_v3_0203_ha/`:

| Directory  | Contents                                                      |
|------------|---------------------------------------------------------------|
| `robot/`   | Robot motions with hand-action + table pose                   |
| `objects/` | Object motions, contact points filtered to ≥ lift frame       |
| `meta/`    | Per-motion meta (table pose/quat/size, object name)           |

And a BPS encoding at `data/motion_lib/benchmark_v3_0203/bps/` (multi-object
datasets only).

### Individual stages

Each stage is a plain Python CLI and can run in isolation.

```bash
# Stage 1 — retarget SMPL-X → G1
python -m grail.retargeting.retarget \
    --data_dir data/genhoi/benchmark_v3/generation/4dhoi_recon_valid/Hunyuan \
    --all --robot unitree_g1 --no_viewer \
    --output_dir data/motion_lib/benchmark_v3_0203

# Stage 2 — hand-action + table-geometry processing
python -m grail.retargeting.process \
    --input  data/motion_lib/benchmark_v3_0203 \
    --output data/motion_lib/benchmark_v3_0203_ha \
    --meta_pkl grail/retargeting/data/g1_skeleton_meta.pkl \
    --include_contact_points --grasp_from_lift \
    --lift_threshold 0.02 --grasp_anticipation_frames 10 \
    --skip_no_lift --per_object

# Add --treat_hands_equally to preserve both arms and derive left/right
# hand actions symmetrically from each hand's contacts.

# Stage 3 — BPS shape encoding (multi-object datasets only)
python -m grail.retargeting.compute_bps \
    --object_usd_dir data/SBTO/pickCylinder/grasp_cylinder/usd\
    --output_dir     data/SBTO/pickCylinder/grasp_cylinder/bps
```

The shell wrappers under
{src}`grail/retargeting/scripts/`
(`retarget.sh`, `process.sh`, `compute_bps.sh`) are thin convenience layers on
top of these CLIs — read them if you want to know the exact defaults.

(terrain-sitting-data)=
### Terrain / sitting data

Terrain (curbs, slopes, stairs) and sitting data involve whole-body interaction
with large environmental objects, not hand-held manipulation. Use
`--zero_out_wrist` to skip hand IK:

```bash
bash grail/retargeting/scripts/retarget.sh \
    data/genhoi/results_terrain_v6/generation/4dhoi_recon_valid/Hunyuan \
    terrain_v6 --zero_out_wrist
```

Then skip the `process.sh` step — terrain data does not need hand-action
preprocessing.

## How the pipeline works

1. **GMR (General Motion Retargeting)** — SMPL-X body model → Unitree G1 MJCF
   via inverse kinematics. The retarget engine lives in
   {src}`imports/GMR` with NVIDIA overrides applied by the install
   script (see {blob}`override README <grail/retargeting/gmr_overrides/README.md>`).
2. **Object mesh → USD** — `convert_mesh.py` runs IsaacLab's `MeshConverter`
   headlessly to produce simulation-ready USD assets with convex-hull collision.
3. **Hand-action + table geometry** — `process.py` derives hand open/close
   commands from object lift/contact timing and applies table geometry fixes.
   By default, the legacy right-hand pickup path zeroes the left arm and keeps
   `hand_action_left` open. Use `--treat_hands_equally` to preserve both arms
   and derive each hand action from that hand's contact timing.
4. **BPS encoding** — `compute_bps.py` samples surface points from each object
   USD and projects them onto a fixed basis-point set, producing a 10-D object
   shape embedding used as a policy observation in
   {blob}`pnp_table <imports/SONIC/gear_sonic/config/exp/manager/universal_token/hoi/pnp_table.yaml>`.

## Troubleshooting

| Symptom                                          | Likely cause / fix                                                                                    |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `ModuleNotFoundError: general_motion_retargeting` | Rerun `bash scripts/setup/install_env_sonic.sh` — it installs GMR editable in the active env.         |
| `ModuleNotFoundError: pxr`                        | `pip install usd-core` (standalone PXR; Isaac Sim-vendored `pxr` is only importable inside kit apps). |
| Black mujoco viewer / `glfwInit failed`           | `export DISPLAY=:1` **before** activating conda (it is an env var, not a conda setting).              |
| Retarget skips motions as "no lift"              | `process.py` rejects motions where the object never rises 2 cm. Override with `--lift_threshold 0.01`. |
| Stale GMR overrides after `git pull`             | Rerun the install script — `rsync` step re-applies `grail/retargeting/gmr_overrides/` idempotently.   |
