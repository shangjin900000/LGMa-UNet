#!/usr/bin/env bash
# Train LGMa-UNet on PCN
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=13200 main.py \
  --launcher pytorch --sync_bn \
  --config ./cfgs/PCN_models/UpTrans_PCN.yaml \
  --exp_name PCN \
  --num_workers 4 --val_freq 10 --val_interval 50
