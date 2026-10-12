#!/usr/bin/env bash
# KITTI: train AdaPoinTr on PCN Cars; optional --start_ckpts from PCN AdaPoinTr
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=13213 main.py \
  --launcher pytorch --sync_bn \
  --config ./cfgs/KITTI_models/AdaPoinTr.yaml \
  --exp_name AdaPoinTr_KITTI \
  --num_workers 4 --val_freq 10 --val_interval 100
