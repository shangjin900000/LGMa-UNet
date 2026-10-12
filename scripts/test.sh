#!/usr/bin/env bash
# Generic test helper
# Usage:
#   bash scripts/test.sh <config.yaml> <ckpt.pth> [exp_name] [mode]
# mode (ShapeNet only): easy | median | hard
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:?config yaml}"
CKPT="${2:?checkpoint}"
EXP="${3:-test}"
MODE_ARG=()
if [[ -n "${4:-}" ]]; then MODE_ARG=(--mode "$4"); fi

CUDA_VISIBLE_DEVICES=0 python main.py --test \
  --config "$CFG" \
  --ckpts "$CKPT" \
  --exp_name "$EXP" \
  "${MODE_ARG[@]}"
