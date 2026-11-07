#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 MiniMind 数据集（默认只下 mini 版两个文件，约 2.8GB）。

用法：
  python scripts/download_data.py --mini          # 只下 pretrain_t2t_mini + sft_t2t_mini
  python scripts/download_data.py --check         # 只检查本地是否齐全
  python scripts/download_data.py --full          # 下全量版（10GB + 14GB，谨慎）
  python scripts/download_data.py --mini --source hf
  python scripts/download_data.py --mini --file pretrain_t2t_mini.jsonl

数据来源（作者托管，二选一）：
  ModelScope : gongjy/minimind_dataset
  HuggingFace: jingyaogong/minimind_dataset
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"

MS_ID = "gongjy/minimind_dataset"
HF_ID = "jingyaogong/minimind_dataset"

MINI_FILES = ["pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl"]
FULL_FILES = ["pretrain_t2t.jsonl", "sft_t2t.jsonl"]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def report(files: list[str]) -> bool:
    ok = True
    print(f"数据目录：{DATASET_DIR}\n")
    for name in files:
        p = DATASET_DIR / name
        if p.exists():
            print(f"  [ok]   {name:<28} {human(p.stat().st_size)}")
        else:
            print(f"  [miss] {name:<28} -")
            ok = False
    return ok


def download_modelscope(files: list[str], local_dir: Path) -> None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        sys.exit("[!] 没装 modelscope：pip install modelscope")

    snapshot_download(
        MS_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=files,
    )


def download_hf(files: list[str], local_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("[!] 没装 huggingface_hub：pip install huggingface_hub")

    for name in files:
        hf_hub_download(
            repo_id=HF_ID,
            repo_type="dataset",
            filename=name,
            local_dir=str(local_dir),
        )


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--mini", action="store_true", help="下载 mini 版（默认）")
    g.add_argument("--full", action="store_true", help="下载全量版（10GB+14GB）")
    g.add_argument("--check", action="store_true", help="只检查本地文件")
    p.add_argument("--source", choices=["modelscope", "hf"], default="modelscope")
    p.add_argument("--file", action="append", help="只下指定文件，可重复")
    args = p.parse_args()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    files = FULL_FILES if args.full else MINI_FILES
    if args.file:
        files = args.file

    if args.check:
        sys.exit(0 if report(files) else 1)

    if report(files):
        print("\n文件已齐全，跳过下载。")
        return

    print(f"\n开始下载（{args.source}）：{', '.join(files)}\n")
    if args.source == "hf":
        download_hf(files, DATASET_DIR)
    else:
        download_modelscope(files, DATASET_DIR)

    print()
    if not report(files):
        sys.exit("\n[!] 下载后仍有文件缺失，请检查上面的报错。")


if __name__ == "__main__":
    main()
