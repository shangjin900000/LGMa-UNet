#!/usr/bin/env bash
# Train AdaPoinTr + SAP (DMC-style) on PCN
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=14210 main.py \
  --launcher pytorch --sync_bn \
  --config ./cfgs/PCN_models/AdaPoinTrSAP.yaml \
  --exp_name AdaPoinTrSAP_PCN \
  --num_workers 4 --val_freq 10 --val_interval 50
