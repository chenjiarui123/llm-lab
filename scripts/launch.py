#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 configs/*.yaml 翻译成 MiniMind 训练脚本的命令行参数并启动。

为什么要这一层：
  MiniMind 的 trainer/*.py 只吃命令行参数，而且内部有若干写死的相对路径：
    - tokenizer_path = '../model'   （相对 cwd，必须是 minimind/trainer）
    - lm_checkpoint  = '../checkpoints'
  所以不能在 llm-lab 根目录直接调用。这里统一：
    1. 读 yaml，映射成 --key value
    2. 把 dataset / save 目录换算成绝对路径
    3. chdir 到 third_party/minimind/trainer 后启动子进程

用法：
  python scripts/launch.py pretrain --config configs/local_8gb.yaml
  python scripts/launch.py sft      --config configs/local_8gb.yaml
  python scripts/launch.py pretrain --config configs/local_8gb.yaml --dry-run
  python scripts/launch.py pretrain --config configs/local_8gb.yaml --extra --save_interval 999999
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STAGE_SCRIPT = {
    "pretrain": "train_pretrain.py",
    "sft": "train_full_sft.py",
}

# yaml 里的 key -> MiniMind 命令行参数名（两边同名，保持直白）
PASSTHROUGH = [
    "batch_size",
    "accumulation_steps",
    "max_seq_len",
    "num_workers",
    "epochs",
    "learning_rate",
    "dtype",
    "log_interval",
    "save_interval",
    "from_weight",
]


def load_yaml(path: Path) -> dict:
    """优先用 PyYAML；没装就退回内置的最小解析器（只支持本项目用到的子集）。"""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml(text)


def _mini_yaml(text: str) -> dict:
    """极简 YAML 子集解析：缩进嵌套 dict、key: value、# 注释、int/float/bool/str。"""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            node: dict = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _cast(value)
    return root


def _cast(v: str):
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    v = v.strip().strip('"').strip("'")
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _stage_from_weight(sec: dict, save_dir: Path, model: dict, minimind: Path, dry_run: bool = False) -> None:
    """MiniMind 的 init_model 默认从 '../out'（相对 cwd=trainer）读初始权重，而 llm-lab 把
    训练产出集中到 out/。这里在启动前把 from_weight 指定的权重硬链/复制到 MiniMind 期望的
    third_party/minimind/out/，使 pretrain/sft/续训等各 stage 通用，无需手改 third_party。
    """
    from_weight = sec.get("from_weight")
    if from_weight in (None, "", "none"):
        return
    hidden = model.get("hidden_size", 768)
    suffix = "_moe" if bool(model.get("use_moe", 0)) else ""
    src = save_dir / f"{from_weight}_{hidden}{suffix}.pth"
    if not src.exists():
        sys.exit(
            f"[!] 找不到初始权重 {src}\n"
            f"    from_weight={from_weight}：先跑对应阶段（如 pretrain）生成该权重"
        )
    dst_dir = minimind / "out"
    dst = dst_dir / src.name
    if dry_run:
        print(f"[stage] 将把 {src} 链到 {dst}（MiniMind init_model 默认读 '../out'）")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
            print(f"[stage] 初始权重已就位：{dst}")
            return
        dst.unlink()
    try:
        os.link(src, dst)
        print(f"[stage] 已硬链接初始权重 -> {dst}")
    except OSError:
        shutil.copyfile(src, dst)
        print(f"[stage] 已复制初始权重 -> {dst}")


def build_argv(stage: str, cfg: dict, extra: list[str], dry_run: bool = False) -> tuple[Path, list[str]]:
    minimind = (ROOT / cfg["paths"]["minimind_dir"]).resolve()
    trainer_dir = minimind / "trainer"
    script = trainer_dir / STAGE_SCRIPT[stage]

    if not script.exists():
        sys.exit(
            f"[!] 找不到 {script}\n"
            f"    先跑 scripts/setup_*.{'ps1' if os.name == 'nt' else 'sh'} 把 MiniMind clone 到 "
            f"{minimind}"
        )

    dataset_dir = (ROOT / cfg["paths"]["dataset_dir"]).resolve()
    save_dir = (ROOT / cfg["paths"]["save_dir"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    sec = cfg[stage]
    data_path = dataset_dir / str(sec["data_file"])
    if not data_path.exists():
        sys.exit(
            f"[!] 找不到数据 {data_path}\n"
            f"    先跑 python scripts/download_data.py --mini"
        )

    model = cfg.get("model", {})
    device = cfg.get("device", "cuda")

    argv = [
        sys.executable,
        str(script),
        "--data_path", str(data_path),
        "--save_dir", str(save_dir),
        "--device", str(device),
        "--hidden_size", str(model.get("hidden_size", 768)),
        "--num_hidden_layers", str(model.get("num_hidden_layers", 8)),
        "--use_moe", str(model.get("use_moe", 0)),
    ]
    for key in PASSTHROUGH:
        if key in sec and sec[key] is not None:
            argv += [f"--{key}", str(sec[key])]

    # 修正 from_weight 权重路径：MiniMind 的 init_model 默认读 trainer/../out
    _stage_from_weight(sec, save_dir, model, minimind, dry_run=dry_run)

    argv += extra
    return trainer_dir, argv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=sorted(STAGE_SCRIPT))
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="追加给 MiniMind 的原始参数，放在最后，例如 --extra --use_wandb",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        sys.exit(f"[!] 配置文件不存在：{cfg_path}")

    cfg = load_yaml(cfg_path)
    cwd, argv = build_argv(args.stage, cfg, args.extra, dry_run=args.dry_run)

    printable = " ".join(argv)
    print(f"[cfg ] {cfg_path}")
    print(f"[cwd ] {cwd}")
    print(f"[cmd ] {printable}\n")

    if args.dry_run:
        return

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Windows 下避免 pyarrow/torch 的 DLL 冲突（MiniMind issue #771 提到的同类问题）
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    raise SystemExit(subprocess.call(argv, cwd=str(cwd), env=env))


if __name__ == "__main__":
    main()
