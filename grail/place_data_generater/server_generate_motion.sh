#!/usr/bin/env bash
# 在 yuguanchao_kimodo 中运行，使用 GPU 1 批量生成 manifest 中的全部动作。
# 用法：bash server_generate_motion.sh [BATCH_SIZE]；默认每批 16 条。
set -eo pipefail
batch_size=${1:-${BATCH_SIZE:-32}}
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo 'BATCH_SIZE 必须是正整数' >&2; exit 2; }

cd /workspace/kimodo
source /workspace/kimodo/deployment/env.sh
set -u
# env.sh 设置 GPU 0，必须在 source 之后覆盖。
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TEXT_ENCODER_MODE=local
export TEXT_ENCODER_DTYPE=float32
export TEXT_ENCODER_DEVICE=cpu
shared=/workspace/kimodo/data/wostep/place_walk
mkdir -p "$shared/motions"

echo "[2/3] 使用 GPU 1 batch 生成动作，每批 $batch_size 条"
python -u - "$shared/manifest.json" "$batch_size" <<'PY'
import json
from pathlib import Path
import shlex
import sys

import torch
from tqdm import tqdm

from kimodo import load_model
from kimodo.constraints import load_constraints_lst
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.postprocess import post_process_motion
from kimodo.scripts import generate as cli
from kimodo.tools import seed_everything

manifest, batch_size = sys.argv[1], int(sys.argv[2])
with open(manifest) as file:
    records = json.load(file)["records"]
if not records:
    raise SystemExit("No constraints in manifest")
if not torch.cuda.is_available():
    raise SystemExit("GPU 1 is not available to Kimodo")

total = len(records)
num_batches = (total + batch_size - 1) // batch_size
print(f"[2/3] 加载模型一次；共 {total} 个文件、{num_batches} 个批次，每批 {batch_size} 条", flush=True)
# CUDA_VISIBLE_DEVICES=1 后，进程内的 cuda:0 对应 GPU 1。
model = load_model("Kimodo-G1-RP-v1", device="cuda:0", text_encoder_fp32=True)
converter = MujocoQposConverter(model.skeleton)

for start in range(0, total, batch_size):
    batch = records[start:start + batch_size]
    batch_index = start // batch_size + 1
    label = f"[2/3] [批次 {batch_index}/{num_batches}] [文件 {start + 1}-{start + len(batch)}/{total}]"
    jobs, batch_constraints = [], []
    print(f"{label} 加载约束", flush=True)
    for index, record in enumerate(batch, start + 1):
        argv = shlex.split(record["command"])
        if argv[:3] != ["python", "-m", "kimodo.scripts.generate"]:
            raise SystemExit("Unexpected Kimodo generation command")
        sys.argv = ["generate", *argv[3:]]
        args = cli.parse_args()
        inputs = cli.get_generation_inputs(args, model.fps)
        if (args.model != "Kimodo-G1-RP-v1" or len(inputs["texts"]) != 1
                or inputs["num_samples"] != 1
                or inputs["num_frames"] != [record["num_frames"]]
                or model.fps != record["model_fps"]):
            raise SystemExit(f"Expected one G1 prompt and matching frame count: {record['name']}")
        cfg_kwargs = cli.resolve_cfg_kwargs(args, inputs.get("meta"))
        jobs.append((record, args, inputs, cfg_kwargs))
        # 与现有 CLI 一致，约束保留在 CPU；模型内部负责移到生成设备。
        batch_constraints.append(load_constraints_lst(inputs["constraints_path"], model.skeleton))
        print(f"[2/3] [{index}/{total}] 已加载约束：{record['name']}", flush=True)

    inputs, cfg_kwargs = jobs[0][2:]
    settings = (inputs["diffusion_steps"], inputs["seed"], cfg_kwargs)
    if any((job[2]["diffusion_steps"], job[2]["seed"], job[3]) != settings for job in jobs):
        raise SystemExit("Each batch must use the same diffusion settings and seed")
    if inputs["seed"] is not None:
        # 与官方 batch 示例一致：每个批次设置一次 seed。
        seed_everything(inputs["seed"])

    print(f"{label} 开始批量生成（GPU 1）", flush=True)
    with torch.inference_mode():
        output = model(
            prompts=[job[2]["texts"][0] for job in jobs],
            num_frames=[record["num_frames"] for record in batch],
            constraint_lst=batch_constraints,
            num_denoising_steps=inputs["diffusion_steps"],
            multi_prompt=False,
            post_processing=False,
            return_numpy=False,
            progress_bar=lambda steps: tqdm(steps, desc=f"{label} 去噪", file=sys.stdout,
                                            mininterval=1.0, disable=False),
            **cfg_kwargs,
        )

    # 先裁掉 batch 填充帧，再逐条进行已有 G1 后处理和导出。
    for local_index, (record, args, _, _) in enumerate(jobs):
        index = start + local_index + 1
        frames = record["num_frames"]
        sample = {key: value[local_index:local_index + 1, :frames] for key, value in output.items()}
        with torch.inference_mode():
            if not args.no_postprocess:
                print(f"[2/3] [{index}/{total}] G1 后处理：{record['name']}", flush=True)
                sample.update(post_process_motion(
                    sample["local_rot_mats"], sample["root_positions"], sample["foot_contacts"],
                    model.skeleton, batch_constraints[local_index],
                ))
            qpos = converter.dict_to_qpos(sample, device=model.device)
        if qpos.shape != (1, frames, 36):
            raise SystemExit(f"Unexpected G1 CSV shape for {record['name']}: {qpos.shape}")
        stem = Path(args.output)
        stem.parent.mkdir(parents=True, exist_ok=True)
        converter.save_csv(qpos[0], f"{stem}.csv")
        save_kimodo_npz(f"{stem}.npz", {key: value[0] for key, value in sample.items()})
        print(f"[2/3] [{index}/{total}] 已保存 CSV/NPZ：{record['name']}（{frames} 帧）", flush=True)
    del output, sample, qpos
    print(f"{label} 完成", flush=True)

print(f"[2/3] 动作生成完成：{total}/{total}", flush=True)
PY
