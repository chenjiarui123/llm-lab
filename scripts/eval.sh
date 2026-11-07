#!/usr/bin/env bash
# 推理测试。用法：bash scripts/eval.sh [full_sft|pretrain] [额外参数...]
set -euo pipefail

WEIGHT="${1:-full_sft}"
shift || true

cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8

MM="third_party/minimind"

if [ ! -f "${MM}/eval_llm.py" ]; then
  echo "[!] 找不到 ${MM}/eval_llm.py，先跑 setup 脚本 clone MiniMind" >&2
  exit 1
fi

if [ ! -f "out/${WEIGHT}_768.pth" ]; then
  echo "[!] 找不到权重 out/${WEIGHT}_768.pth" >&2
  echo "    先跑 bash scripts/train_pretrain.sh / train_sft.sh" >&2
  exit 1
fi

# eval_llm.py 用 './{save_dir}' 拼权重路径，所以必须在 llm-lab 根目录执行，
# 并把 tokenizer 目录显式指到 MiniMind 的 model/。
python "${MM}/eval_llm.py" \
  --load_from "${MM}/model" \
  --save_dir out \
  --weight "${WEIGHT}" \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  "$@"
