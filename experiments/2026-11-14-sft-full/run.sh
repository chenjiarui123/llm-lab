#!/usr/bin/env bash
# 复现 2026-11-14 SFT 全量训练（AutoDL 单卡 32GB）。
# 用法： bash experiments/2026-11-14-sft-full/run.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

bash scripts/train_sft.sh experiments/2026-11-14-sft-full/config.yaml \
  > experiments/2026-11-14-sft-full/train.log 2>&1
rc=$?
echo "=== FINISHED rc=${rc} at $(date) ===" >> experiments/2026-11-14-sft-full/train.log
exit $rc
