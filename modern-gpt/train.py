"""在 Shakespeare 字符级语料上训练 ModernGPT 实验模型（中文注释版）。

示例
----
先跑 baseline：
    python train.py --max-iters 300

每次只打开一个现代组件：
    python train.py --max-iters 300 --use-rmsnorm
    python train.py --max-iters 300 --use-rmsnorm --use-swiglu
    python train.py --max-iters 300 --use-rmsnorm --use-swiglu --use-rope
    python train.py --max-iters 300 --use-rmsnorm --use-swiglu --use-rope --n-kv-head 2
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch.nn import functional as F

from model import GPTConfig, ModernGPT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "ng-video-lecture", "input.txt"),
    )
    p.add_argument("--max-iters", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 现代架构开关
    p.add_argument("--use-rmsnorm", action="store_true")
    p.add_argument("--use-rope", action="store_true")
    p.add_argument("--use-swiglu", action="store_true")
    p.add_argument("--n-kv-head", type=int, default=None)
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--tie-weights", action="store_true")
    p.add_argument("--rope-theta", type=float, default=10000.0)

    # 小配置，方便在 CPU 上快速调试。
    p.add_argument("--tiny", action="store_true")
    return p.parse_args()


def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到数据文件：{path}\n"
            "请在 modern-gpt/ 目录下运行，或用 --data ../ng-video-lecture/input.txt 指定路径"
        )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return text, chars, stoi, itos, data


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data[split], block_size, batch_size, device)
            _, loss, _ = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    if args.tiny:
        args.n_layer = 2
        args.n_head = 4
        args.n_embd = 128
        args.block_size = 64
        args.batch_size = 16
        args.max_iters = min(args.max_iters, 100)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    text, chars, stoi, itos, data = load_data(args.data)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    config = GPTConfig(
        vocab_size=len(chars),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        use_rmsnorm=args.use_rmsnorm,
        use_rope=args.use_rope,
        use_swiglu=args.use_swiglu,
        n_kv_head=args.n_kv_head,
        use_cache=args.use_cache,
        tie_weights=args.tie_weights,
        rope_theta=args.rope_theta,
    )
    if config.n_kv_head is not None and config.n_head % config.n_kv_head != 0:
        raise ValueError("--n-kv-head 必须能整除 --n-head")

    model = ModernGPT(config).to(args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device: {args.device}")
    print(
        "switches: "
        f"rmsnorm={config.use_rmsnorm}, rope={config.use_rope}, "
        f"swiglu={config.use_swiglu}, kv_heads={config.n_kv_head or config.n_head}"
    )
    print(f"parameters: {n_params:.3f}M")
    print(f"vocab size: {len(chars)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start = time.time()
    for it in range(args.max_iters):
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(
                model,
                {"train": train_data, "val": val_data},
                args.block_size,
                args.batch_size,
                args.device,
                args.eval_iters,
            )
            elapsed = time.time() - start
            print(
                f"step {it:4d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {elapsed:.1f}s"
            )

        xb, yb = get_batch(train_data, args.block_size, args.batch_size, args.device)
        logits, loss, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=args.device)
    generated = model.generate(context, max_new_tokens=200)
    print("\n--- sample ---")
    print("".join(itos[i] for i in generated[0].tolist()))


if __name__ == "__main__":
    main()
