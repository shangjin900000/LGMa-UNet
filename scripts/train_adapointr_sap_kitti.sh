#!/usr/bin/env bash
# KITTI: train AdaPoinTrSAP on PCN Cars
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=14213 main.py \
  --launcher pytorch --sync_bn \
  --config ./cfgs/KITTI_models/AdaPoinTrSAP.yaml \
  --exp_name AdaPoinTrSAP_KITTI \
  --num_workers 4 --val_freq 10 --val_interval 100
