#!/usr/bin/env bash
# KITTI: train on PCN Cars (with random_dropping), test on KITTI
# Optional: add --start_ckpts /path/to/PCN/ckpt-best.pth to init from PCN weights
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=13204 main.py \
  --launcher pytorch --sync_bn \
  --config ./cfgs/KITTI_models/UpTrans_KITTI.yaml \
  --exp_name KITTI \
  --num_workers 4 --val_freq 10 --val_interval 100
