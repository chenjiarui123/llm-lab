#!/usr/bin/env bash
# Linux / AutoDL 环境初始化
# 用法： bash scripts/setup_autodl.sh
#        REINSTALL=1 bash scripts/setup_autodl.sh   # 强制重建 venv
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "==> 工作目录: ${ROOT}"

PY=${PY:-python3}
command -v "$PY" >/dev/null || { echo "[!] 找不到 python3"; exit 1; }
"$PY" --version

# ---------- 1. 系统依赖（AutoDL 镜像一般已内置）----------
if command -v apt-get >/dev/null 2>&1; then
  need=""
  for p in git wget; do command -v "$p" >/dev/null || need="$need $p"; done
  if [ -n "$need" ]; then
    echo "==> apt-get install$need"
    apt-get update -qq && apt-get install -y -qq $need
  fi
fi

# ---------- 2. venv（有系统 torch 就复用它）----------
if [ "${REINSTALL:-0}" = "1" ] && [ -d .venv ]; then
  echo "==> 删除旧 venv"
  rm -rf .venv
fi

if [ ! -d .venv ]; then
  echo "==> 创建 venv (--system-site-packages)"
  "$PY" -m venv .venv --system-site-packages
fi

VPY="${ROOT}/.venv/bin/python"
"$VPY" -m pip install --upgrade pip -q

echo "==> 安装 requirements/base.txt"
"$VPY" -m pip install -r requirements/base.txt

echo "==> 安装 requirements/train.txt"
"$VPY" -m pip install -r requirements/train.txt

# ---------- 3. 检查 torch ----------
echo "==> 检查 torch"
"$VPY" - <<'PY'
import torch
print('torch      :', torch.__version__)
print('cuda avail :', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('gpu        :', p.name)
    print('vram       :', round(p.total_memory / 1024 ** 3, 1), 'GB')
    print('bf16       :', torch.cuda.is_bf16_supported())
else:
    print('[!] CUDA 不可用：确认镜像带了 GPU 版 torch，或 pip install torch --index-url https://download.pytorch.org/whl/cu121')
PY

# ---------- 4. clone MiniMind ----------
if [ ! -d third_party/minimind ]; then
  echo "==> clone MiniMind -> third_party/minimind"
  mkdir -p third_party
  git clone --depth 1 https://github.com/jingyaogong/minimind third_party/minimind
else
  echo "==> third_party/minimind 已存在，跳过 clone"
fi

COMMIT=$(git -C third_party/minimind rev-parse --short HEAD)
echo "==> MiniMind commit: ${COMMIT}"

# ---------- 5. 目录 ----------
mkdir -p dataset out

echo ""
echo "完成。下一步："
echo "  source .venv/bin/activate"
echo "  python scripts/download_data.py --mini"
echo "  bash scripts/train_pretrain.sh configs/autodl_24gb.yaml"
