#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=13212 main.py \
  --launcher pytorch --sync_bn \
  --config ./cfgs/ShapeNet34_models/AdaPoinTr.yaml \
  --exp_name AdaPoinTr_ShapeNet34 \
  --num_workers 4 --val_freq 10 --val_interval 100
