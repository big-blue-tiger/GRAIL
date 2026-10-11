#!/usr/bin/env bash
# 在 yuguanchao_sonic 中运行；可传入 PKL 匹配模式，默认处理全部动作。
# 用法：bash server_generate_constraints.sh ["pickup_table__alcohol_4__000.pkl"]
set -euo pipefail

cd /home/GRAIL
dataset=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat
shared=/home/kimodo/data/wostep/place_walk_1010

# /home/kimodo 与 Kimodo 容器的 /workspace/kimodo 共享同一宿主机目录。
/isaac-sim/python.sh -m grail.place_data_generater.constraints \
  --dataset-dir "$dataset" \
  --output-dir "$shared" \
  --pattern "${1:-*.pkl}" \
  --kimodo-repo /home/kimodo \
  --remote-dir /workspace/kimodo/data/wostep/place_walk_1010
