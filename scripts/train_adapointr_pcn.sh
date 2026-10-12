#!/usr/bin/env bash
# Train original AdaPoinTr (PoinTr) on PCN
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=13210 main.py \
  --launcher pytorch --sync_bn \
  --config ./cfgs/PCN_models/AdaPoinTr.yaml \
  --exp_name AdaPoinTr_PCN \
  --num_workers 4 --val_freq 10 --val_interval 50
