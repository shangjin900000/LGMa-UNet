#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=14211 main.py \
  --launcher pytorch --sync_bn \
  --config ./cfgs/ShapeNet55_models/AdaPoinTrSAP.yaml \
  --exp_name AdaPoinTrSAP_ShapeNet55 \
  --num_workers 4 --val_freq 10 --val_interval 100
