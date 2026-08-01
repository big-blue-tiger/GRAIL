#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SONIC_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

GPU="${GPU:-0}"
DAGGER_ROUNDS="${DAGGER_ROUNDS:-100}"
EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SCHEDULER_NUM_EPOCHS="${SCHEDULER_NUM_EPOCHS:-500}"

RL_CHECKPOINT="${RL_CHECKPOINT:-${SONIC_ROOT}/logs_rl/pnp_table_pnp_table-721/last.pt}"
MOTION_INPUT="${MOTION_INPUT:-${SONIC_ROOT}/../../../sbto/datas/sbto_to_grail/pickup_table/robot}"
DATASET_PATHS="${DATASET_PATHS:-[../outputs/**/**/*.object_aware.pkl]}"
DAGGER_ROOT="${SONIC_ROOT}/outputs/dagger"
DAGGER_OUTPUT="${DAGGER_ROOT}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/data/outputs/dagger_automation/$(date +%Y.%m.%d-%H.%M.%S)-$$}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
  requested_python="${PYTHON_BIN}"
  PYTHON_BIN="$(command -v -- "${requested_python}" 2>/dev/null)" ||
    die "PYTHON_BIN is not executable: ${requested_python}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif [[ -x /workspace/isaaclab/_isaac_sim/python.sh ]]; then
  PYTHON_BIN=/workspace/isaaclab/_isaac_sim/python.sh
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  die "Python not found; set PYTHON_BIN to a Python executable"
fi

reset_dagger_data() {
  [[ "${DAGGER_ROOT}" == "${SONIC_ROOT}/outputs/dagger" ]] ||
    die "refusing to delete unexpected directory: ${DAGGER_ROOT}"
  rm -rf -- "${DAGGER_ROOT}"
  mkdir -p -- "${DAGGER_OUTPUT}"
}

latest_checkpoint() {
  local checkpoint_dir="$1"
  if [[ -s "${checkpoint_dir}/latest.ckpt" ]]; then
    printf '%s\n' "${checkpoint_dir}/latest.ckpt"
    return
  fi
  local newest=""
  local path
  while IFS= read -r -d '' path; do
    if [[ -z "${newest}" || "${path}" -nt "${newest}" ]]; then
      newest="${path}"
    fi
  done < <(find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.ckpt' -print0)
  [[ -n "${newest}" && -s "${newest}" ]] ||
    die "no checkpoint produced in ${checkpoint_dir}"
  printf '%s\n' "${newest}"
}

train_generator() {
  local round="$1"
  local num_epochs="$2"
  local start_checkpoint="${3:-}"
  local run_dir
  [[ "${num_epochs}" =~ ^[1-9][0-9]*$ ]] ||
    die "train_generator num_epochs must be a positive integer"
  run_dir="${RUN_ROOT}/round_$(printf '%02d' "${round}")"
  mkdir -p -- "${run_dir}"

  local command=(
    "${PYTHON_BIN}" sugar_il/workspace/train_generator_workspace.py
    task=ObjectAware
    "task.dataset_paths=${DATASET_PATHS}"
    "num_epochs=${num_epochs}"
    training.val_every=1
    "training.scheduler_num_epochs=${SCHEDULER_NUM_EPOCHS}"
    checkpoint.topk.monitor_key=val_loss
    'checkpoint.topk.format_str="epoch-{epoch:04d}-val_loss-{val_loss:.3f}.ckpt"'
    checkpoint.save_last_ckpt=true
    "log_path=${run_dir}"
  )
  if [[ -n "${start_checkpoint}" ]]; then
    command+=(
      "start_ckpt_path=${start_checkpoint}"
      training.resume=true
    )
  fi

  echo "==> Training round ${round} (${num_epochs} epochs)"
  (
    cd -- "${SCRIPT_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${command[@]}"
  )
  FLOW_CHECKPOINT="$(latest_checkpoint "${run_dir}/checkpoints")"
  echo "==> Selected checkpoint: ${FLOW_CHECKPOINT}"
}

generate_dagger_data() {
  local round="$1"
  local generator_checkpoint="$2"
  reset_dagger_data
  echo "==> Generating DAgger round ${round} with ${generator_checkpoint}"
  (
    cd -- "${SONIC_ROOT}"
    "${PYTHON_BIN}" -u sugar_il/sugar_il/workspace/get_generator_data_for_dagger.py \
      --gpu "${GPU}" \
      --checkpoint "${RL_CHECKPOINT}" \
      --generator-checkpoint "${generator_checkpoint}" \
      --input "${MOTION_INPUT}" \
      --output-dir "${DAGGER_OUTPUT}" \
      --no-video-rendering \
      --batch-size "${BATCH_SIZE}"
  )
}

[[ "${DAGGER_ROUNDS}" =~ ^[1-9][0-9]*$ ]] ||
  die "DAGGER_ROUNDS must be a positive integer"
[[ "${EPOCHS_PER_ROUND}" =~ ^[1-9][0-9]*$ ]] ||
  die "EPOCHS_PER_ROUND must be a positive integer"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] ||
  die "BATCH_SIZE must be a positive integer"
if [[ -z "${SCHEDULER_NUM_EPOCHS}" ]]; then
  # Later rounds contain GRAIL + roughly one GRAIL-sized DAgger dataset.
  SCHEDULER_NUM_EPOCHS=$((EPOCHS_PER_ROUND * (1 + 2 * DAGGER_ROUNDS)))
fi
[[ "${SCHEDULER_NUM_EPOCHS}" =~ ^[1-9][0-9]*$ ]] ||
  die "SCHEDULER_NUM_EPOCHS must be a positive integer"
[[ -f "${RL_CHECKPOINT}" ]] || die "RL checkpoint not found: ${RL_CHECKPOINT}"
[[ -d "${MOTION_INPUT}" ]] || die "motion input not found: ${MOTION_INPUT}"

mkdir -p -- "${RUN_ROOT}"
reset_dagger_data

FLOW_CHECKPOINT=""
train_generator 0 300
flow_checkpoint="${FLOW_CHECKPOINT}"
for ((round = 1; round <= DAGGER_ROUNDS; round++)); do
  generate_dagger_data "${round}" "${flow_checkpoint}"
  train_generator "${round}" "${EPOCHS_PER_ROUND}" "${flow_checkpoint}"
  flow_checkpoint="${FLOW_CHECKPOINT}"
done

echo "DAgger completed ${DAGGER_ROUNDS} round(s)."
echo "Final Flow Matching checkpoint: ${flow_checkpoint}"
echo "Latest DAgger data: ${DAGGER_OUTPUT}"
