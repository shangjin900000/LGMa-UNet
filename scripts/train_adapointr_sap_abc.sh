#!/usr/bin/env bash
# Train AdaPoinTr + SAP on ABC. Generate PSR fields first (S=2.0):
#   python SAP/scripts/process_abc_psr.py --data_root /root/autodl-tmp/datasets/2024_abc_train --scale 2.0
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=14214 main.py \
  --launcher pytorch --sync_bn \
  --config ./cfgs/ABC_models/AdaPoinTrSAP.yaml \
  --exp_name AdaPoinTrSAP_ABC \
  --num_workers 4 --val_freq 10 --val_interval 50
