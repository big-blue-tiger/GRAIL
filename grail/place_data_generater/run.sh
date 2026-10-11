#!/usr/bin/env bash
# One motion: constraints -> server Kimodo -> download -> append -> replay video.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
name=${1:?Usage: run.sh MOTION_NAME [--dry-run] [--skip-render]}
shift
[[ "$name" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] || { echo "Invalid motion name" >&2; exit 2; }
dry_run=false
render=true
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    --skip-render) render=false ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

dataset="$repo/data/hf_dataset/data_update/data/pickup_table_walk_concat"
output="$repo/data/hf_dataset/data_update/data/pickup_table_place_left_right"
constraints="$dataset/kimodo_place_constraints/$name"
downloads="$dataset/kimodo_place"
server=cuixinru@202.120.37.249
remote_dir=/home/cuixinru/data0/kimodo/data/wostep/place_walk
container_dir=/workspace/kimodo/data/wostep/place_walk
container=yuguanchao_kimodo
ssh_args=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p 9989)
rsync_shell='ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p 9989'

set +u
source /home/tide/miniconda3/etc/profile.d/conda.sh
conda activate grail
set -u

echo '[1/5] Export local constraints'
python -m grail.place_data_generater.constraints \
  --dataset-dir "$dataset" --output-dir "$constraints" --pattern "$name.pkl" \
  --remote-dir "$container_dir"
command=$(python - "$constraints/manifest.json" <<'PY'
import json, shlex, sys
records = json.load(open(sys.argv[1]))['records']
if len(records) != 1:
    raise SystemExit('Expected exactly one generated constraint')
argv = shlex.split(records[0]['command'])
if argv[:3] != ['python', '-m', 'kimodo.scripts.generate']:
    raise SystemExit('Unexpected Kimodo generation command')
# Enable G1 post-processing only in this generation process.
runner = '''import inspect
import kimodo.scripts.generate as cli
source = inspect.getsource(cli.main)
disabled = 'use_postprocess = False if "g1" in resolved_model else (not args.no_postprocess)'
if source.count(disabled) != 1:
    raise SystemExit('Kimodo G1 post-processing switch changed; cannot enable it safely')
source = source.replace(disabled, 'use_postprocess = not args.no_postprocess')
exec(compile(source, cli.__file__, 'exec'), cli.__dict__)
print('G1 Post-Processing enabled for this generation', flush=True)
cli.main()
'''
print(shlex.join([argv[0], '-c', runner, *argv[3:]]))
PY
)
echo "$command"
echo "Output: $output/robot/$name.pkl"
if "$dry_run"; then
  echo 'Dry run complete; constraints were written locally, with no server call or final dataset change.'
  exit 0
fi
for tool in ssh rsync; do command -v "$tool" >/dev/null; done

echo '[2/5] Upload and generate on the server'
ssh "${ssh_args[@]}" "$server" /bin/bash -s <<REMOTE
set -euo pipefail
if [ "\$(docker inspect -f '{{.State.Running}}' '$container')" != true ]; then
  docker start '$container'
fi
docker exec '$container' mkdir -p '$container_dir/motions'
if [[ ! -w '$remote_dir' || ! -w '$remote_dir/motions' ]]; then
  docker exec '$container' chown "\$(id -u):\$(id -g)" '$container_dir' '$container_dir/motions'
fi
REMOTE
rsync -av --checksum -e "$rsync_shell" -- "$constraints/$name.json" "$server:$remote_dir/$name.json"
mkdir -p "$downloads/logs"
{
  cat <<'REMOTE'
set -eo pipefail
cd /workspace/kimodo
source /workspace/kimodo/deployment/env.sh
set -u
export HF_HUB_OFFLINE=1
export TEXT_ENCODER_MODE=local
export TEXT_ENCODER_DTYPE=float32
export TEXT_ENCODER_DEVICE=cpu
REMOTE
  printf "rm -f -- '%s/motions/%s.csv' '%s/motions/%s.npz'\n" \
    "$container_dir" "$name" "$container_dir" "$name"
  printf '%s\n' "$command"
  printf "test -s '%s/motions/%s.csv'\n" "$container_dir" "$name"
  printf "test -s '%s/motions/%s.npz'\n" "$container_dir" "$name"
} | ssh "${ssh_args[@]}" "$server" "docker exec -i '$container' /bin/bash -s" \
    2>&1 | tee "$downloads/logs/$name.log"

echo '[3/5] Download generated CSV and NPZ'
work=$(mktemp -d /tmp/grail-place.XXXXXXXX)
trap 'rm -rf -- "$work"' EXIT
rsync -av --checksum -e "$rsync_shell" -- \
  "$server:$remote_dir/motions/$name.csv" \
  "$server:$remote_dir/motions/$name.npz" "$work/"
test -s "$work/$name.csv"
test -s "$work/$name.npz"
python - "$work/$name.csv" "$constraints/manifest.json" <<'PY'
import json, numpy as np, sys
data = np.loadtxt(sys.argv[1], delimiter=',', ndmin=2)
record = json.load(open(sys.argv[2]))['records'][0]
if data.shape != (record['num_frames'], 36) or not np.isfinite(data).all():
    raise SystemExit(f'Invalid downloaded G1 CSV: {data.shape}')
PY
mv -f -- "$work/$name.csv" "$work/$name.npz" "$downloads/"

echo '[4/5] Append locally'
python -m grail.place_data_generater.concat --dataset-dir "$dataset" \
  --output-dir "$output" --manifest "$constraints/manifest.json" \
  --csv "$downloads/$name.csv" --name "$name" --overwrite

if ! "$render"; then
  echo "Done: $output/robot/$name.pkl"
  exit 0
fi
echo '[5/5] Render this motion locally'
mkdir -p "$work/view/robot"
ln -s "$output/robot/$name.pkl" "$work/view/robot/$name.pkl"
for dir in objects object_usd meta; do ln -s "$output/$dir" "$work/view/$dir"; done
CUDA_VISIBLE_DEVICES=0 bash grail/visualization/scripts/visualize.sh \
  "$work/view" 0 1.5,-1.5,1.0 xyzw local 1
  
test -s "$work/view/vis/$name.mp4"
mkdir -p "$output/vis"
mv -f -- "$work/view/vis/$name.mp4" "$output/vis/$name.mp4"
echo "Done: $output/vis/$name.mp4"
