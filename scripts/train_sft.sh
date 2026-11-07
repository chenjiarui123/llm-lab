#!/usr/bin/env bash
# SFT（全参指令微调）。用法：bash scripts/train_sft.sh [configs/xxx.yaml] [额外参数...]
set -euo pipefail

CONFIG="${1:-configs/local_8gb.yaml}"
shift || true

cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== sft | config: ${CONFIG} ==="
echo "    前置条件: out/pretrain_768.pth 已存在（config 里 from_weight: pretrain）"
python scripts/launch.py sft --config "$CONFIG" "$@"
