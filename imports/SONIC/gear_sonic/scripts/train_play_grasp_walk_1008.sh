#!/usr/bin/env bash
# Run from the server's activated Isaac Lab environment.
# Example: bash gear_sonic/scripts/train_play_grasp_walk_1008.sh --delay-hours 2 --delay-minutes 30
set -Eeuo pipefail
trap 'status=$?; printf "\nError at line %s (exit %s): %s\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

usage() {
  cat <<'EOF'
Usage: bash train_play_grasp_walk_1008.sh [--delay-hours N] [--delay-minutes N] [--dry-run]

Train to 15000 iterations, pausing every 3000 iterations to render play videos.
All training segments use the same experiment directory and online W&B run.
The delay defaults to zero. --dry-run prints all commands without waiting or running them.

Optional environment variables:
  GRAIL_ROOT       Repository root (defaults to the repository containing this script)
  PYTHON_BIN       Python executable (default: python, or the container's Isaac Sim python.sh)
  EXPERIMENT_NAME  Experiment name (default: GRASP_WALK_1008)
  CUDA_VISIBLE_DEVICES  GPU selection, inherited by both training and play
EOF
}

delay_hours=0
delay_minutes=0
dry_run=false
while (($#)); do
  case "$1" in
    --delay-hours|--delay-minutes)
      option=$1
      if (($# < 2)) || [[ ! $2 =~ ^[0-9]+$ ]]; then
        printf 'Error: %s requires a nonnegative integer.\n' "$option" >&2
        exit 1
      fi
      if [[ $option == --delay-hours ]]; then
        delay_hours=$((10#$2))
      else
        delay_minutes=$((10#$2))
      fi
      shift 2
      ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 1 ;;
  esac
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
grail_root=${GRAIL_ROOT:-$(cd -- "$script_dir/../../../.." && pwd)}
sonic_dir=$grail_root/imports/SONIC
python_bin=${PYTHON_BIN:-python}
# This container defines `python` as an interactive alias, absent in bash scripts.
if [[ -z ${PYTHON_BIN:-} ]] && ! command -v python >/dev/null 2>&1 \
  && [[ -x /workspace/isaaclab/_isaac_sim/python.sh ]]; then
  python_bin=/workspace/isaaclab/_isaac_sim/python.sh
fi
experiment_name=${EXPERIMENT_NAME:-GRASP_WALK_1008}
dataset_dir=$grail_root/data/hf_dataset/data_update/data/pickup_table_place_concat
asset_root=$sonic_dir/gear_sonic/data/assets/robot_description/mjcf/
teacher_checkpoint=$sonic_dir/models/pnp_table/last.pt
# POSIX timezone works even when the container has no Asia/Shanghai tzdata.
experiment_dir=$sonic_dir/logs_rl/GRAB_Tracking/$experiment_name-$(TZ=CST-8 date +%Y%m%d_%H%M%S)
video_root=$grail_root/outputs/$(basename -- "$experiment_dir")
log_dir=$experiment_dir/train_play_logs
delay_seconds=$((delay_hours * 3600 + delay_minutes * 60))
total_iterations=15000
segment_iterations=3000

cd -- "$sonic_dir"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
# Online mode supports resuming the same run via experiment_dir/meta.yaml.
export WANDB_MODE=online

printf 'Experiment: %s\nVideos: %s\nStart delay: %s hours %s minutes\n' \
  "$experiment_dir" "$video_root" "$delay_hours" "$delay_minutes"

if ! $dry_run; then
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$python_bin" >&2
    printf 'Activate your Isaac Lab environment, or set PYTHON_BIN to its Python launcher.\n' >&2
    exit 1
  fi
  printf 'Python executable: %s\n' "$(command -v "$python_bin")"
  for required in "$dataset_dir/robot" "$dataset_dir/objects" "$dataset_dir/object_usd" \
    "$dataset_dir/bps" "$asset_root" "$teacher_checkpoint"; do
    if [[ ! -e $required ]]; then
      printf 'Missing required path: %s\n' "$required" >&2
      exit 1
    fi
  done
  if [[ -e $experiment_dir ]]; then
    printf 'Experiment directory already exists: %s\n' "$experiment_dir" >&2
    exit 1
  fi
  deadline=$((SECONDS + delay_seconds))
  while ((SECONDS < deadline)); do
    remaining=$((deadline - SECONDS))
    printf '\rStarts in %02d:%02d:%02d  ' \
      "$((remaining / 3600))" "$((remaining / 60 % 60))" "$((remaining % 60))"
    sleep_seconds=60
    if ((remaining < sleep_seconds)); then sleep_seconds=$remaining; fi
    sleep "$sleep_seconds"
  done
  printf '\nStarting train/play loop.\n'
  mkdir -p -- "$log_dir"
fi

run_command() {
  local log_file=$1
  shift
  printf '\nCommand:'
  printf ' %q' "$@"
  printf '\n'
  if ! $dry_run; then
    "$@" 2>&1 | tee "$log_file"
  fi
}

checkpoint=
wandb_id=
for ((target=segment_iterations; target<=total_iterations; target+=segment_iterations)); do
  printf -v step '%06d' "$target"
  resume_args=(resume=False)
  if [[ -n $checkpoint ]]; then
    resume_args=(resume=True "checkpoint=$checkpoint")
  fi
  printf '\nTraining to iteration %s/%s.\n' "$target" "$total_iterations"
  run_command "$log_dir/train_$step.log" "$python_bin" gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/distill/robocasa_pickup_table_place_mlp_decoder_latent_vector_obs_joint_micro_step \
    headless=True num_envs=4096 "experiment_name=$experiment_name" \
    "experiment_dir=$experiment_dir" "${resume_args[@]}" \
    "manager_env.config.object_usd_path=$dataset_dir/object_usd" \
    "manager_env.commands.motion.motion_lib_cfg.motion_file=$dataset_dir/robot" \
    "manager_env.commands.motion.motion_lib_cfg.object_motion_file=$dataset_dir/objects" \
    "manager_env.commands.motion.motion_lib_cfg.bps_dir=$dataset_dir/bps" \
    "manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=$asset_root" \
    "algo.config.teacher_checkpoint=$teacher_checkpoint" \
    "algo.config.num_learning_iterations=$total_iterations" \
    algo.config.ppo_bc_loss_schedule.adaptive_after_iteration=1000 \
    algo.config.ppo_bc_loss_schedule.bc_min_coef=0.04 \
    use_wandb=True "wandb.wandb_dir=$experiment_dir/.wandb" \
    "callbacks.model_save.save_frequency=$segment_iterations" \
    "++callbacks.model_save.save_last_frequency=$segment_iterations" \
    callbacks.model_save.max_disk_usage=null \
    +callbacks.segment_stop._target_=gear_sonic.trl.callbacks.segment_stop_callback.SegmentStopCallback \
    "+callbacks.segment_stop.stop_iteration=$target"

  # This is the newest numbered checkpoint, saved at exactly the segment boundary.
  checkpoint=$experiment_dir/model_step_$step.pt
  if ! $dry_run; then
    if [[ ! -s $checkpoint || ! -s $experiment_dir/last.pt ]]; then
      printf 'Missing segment checkpoint at iteration %s; stopping.\n' "$target" >&2
      exit 1
    fi
    current_wandb_id=$("$python_bin" - "$experiment_dir/meta.yaml" <<'PY'
import sys
import yaml
with open(sys.argv[1]) as stream:
    run_id = yaml.safe_load(stream).get("wandb_run")
if not run_id:
    raise SystemExit("meta.yaml has no W&B run ID; stopping.")
print(run_id)
PY
    )
    if [[ -n $wandb_id && $current_wandb_id != "$wandb_id" ]]; then
      printf 'W&B run ID changed; stopping.\n' >&2
      exit 1
    fi
    wandb_id=$current_wandb_id
    printf 'Continuing W&B run: %s\n' "$wandb_id"
  fi

  video_dir=$video_root/iteration_$step
  printf '\nRendering iteration %s to %s.\n' "$target" "$video_dir"
  run_command "$log_dir/play_$step.log" "$python_bin" gear_sonic/eval_agent_trl.py \
    "+checkpoint=$checkpoint" +headless=True ++num_envs=16 +run_once=True \
    ++use_wandb=False \
    ++manager_env.config.render_results=True \
    "++manager_env.config.save_rendering_dir=$video_dir" \
    '~manager_env/recorders=empty' '+manager_env/recorders=render' \
    "++manager_env.config.object_usd_path=$dataset_dir/object_usd" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$dataset_dir/robot" \
    "++manager_env.commands.motion.motion_lib_cfg.object_motion_file=$dataset_dir/objects" \
    "++manager_env.commands.motion.motion_lib_cfg.bps_dir=$dataset_dir/bps" \
    "++manager_env.commands.motion.motion_lib_cfg.asset.assetRoot=$asset_root" \
    ++manager_env.terminations.ee_body_pos.params.threshold=25 \
    ++manager_env.terminations.object_pos_deviation.params.threshold=25 \
    ++manager_env.terminations.anchor_ori_full.params.threshold=2500 \
    ++manager_env.terminations.place_carry_time_out.params.enabled=False
done

printf '\nFinished %s iterations and all five play runs.\nExperiment: %s\nVideos: %s\n' \
  "$total_iterations" "$experiment_dir" "$video_root"
