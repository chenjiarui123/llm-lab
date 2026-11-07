#!/usr/bin/env bash
# 预训练。用法：bash scripts/train_pretrain.sh [configs/xxx.yaml] [额外参数...]
set -euo pipefail

CONFIG="${1:-configs/local_8gb.yaml}"
shift || true

cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== pretrain | config: ${CONFIG} ==="
python scripts/launch.py pretrain --config "$CONFIG" "$@"
