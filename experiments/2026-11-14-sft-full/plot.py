"""Plot SFT loss / lr from a MiniMind trainer log.

Usage:
  python plot.py [train.log] [loss.png]
默认读同目录下 train.log，输出同目录 loss.png。
"""
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "train.log")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "loss.png")

PAT = re.compile(
    r"\((\d+)/(\d+)\), loss: ([\d.]+), logits_loss: ([\d.]+), "
    r"aux_loss: ([\d.]+), lr: ([\d.eE+-]+)"
)

steps, loss, aux, lr = [], [], [], []
with open(LOG, encoding="utf-8", errors="replace") as f:
    for line in f:
        for part in line.replace("\r", "\n").split("\n"):
            m = PAT.search(part)
            if m:
                steps.append(int(m.group(1)))
                loss.append(float(m.group(3)))
                aux.append(float(m.group(5)))
                lr.append(float(m.group(6)))

if not steps:
    raise SystemExit("no data parsed from " + LOG)

steps = np.asarray(steps, dtype=float)
loss = np.asarray(loss, dtype=float)
aux = np.asarray(aux, dtype=float)
lr = np.asarray(lr, dtype=float)
total = int(steps[-1])


def ema(x, alpha=0.02):
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


sm = ema(loss, 0.02)

BG = "#0d1117"
FG = "#c9d1d9"
DIM = "#6e7681"
GRID = "#21262d"
BLUE = "#79c0ff"
RED = "#ff7b72"


def kfmt(v, _pos):
    v = float(v)
    if v >= 1000:
        s = f"{v / 1000:.1f}k"
        return s.replace(".0k", "k")
    return f"{int(v)}"


def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=15, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=DIM, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(kfmt))
    ax.set_xlabel("Step", color=DIM, fontsize=10, labelpad=6)


fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), facecolor=BG)
fig.subplots_adjust(left=0.045, right=0.985, top=0.86, bottom=0.19, wspace=0.16)

ax = axes[0]
style(ax, "logits_loss")
ax.plot(steps, loss, color=BLUE, lw=0.7, alpha=0.55, label="raw")
ax.plot(steps, sm, color=RED, lw=1.7, label="EMA 0.02")
ax.set_ylim(bottom=0)
ax.annotate(
    f"final {loss[-1]:.3f}   mean(last 20 pts) {loss[-20:].mean():.3f}",
    xy=(1.0, 0.06),
    xycoords="axes fraction",
    ha="right",
    color=DIM,
    fontsize=9,
)
ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=10)

ax = axes[1]
style(ax, "logits_loss  (first 20k, zoom)")
zoom = steps <= 20000
ax.plot(steps[zoom], loss[zoom], color=BLUE, lw=0.8, alpha=0.55, label="raw")
ax.plot(steps[zoom], sm[zoom], color=RED, lw=1.7, label="EMA 0.02")
ax.set_ylim(bottom=0)
ax.annotate(
    f"first 20k steps = {20000 / total * 100:.0f}% of the run, "
    f"{loss[0]:.2f} -> {sm[zoom][-1]:.2f}",
    xy=(0.5, 0.10),
    xycoords="axes fraction",
    ha="center",
    color=DIM,
    fontsize=9,
)

ax = axes[2]
style(ax, "learning_rate")
ax.plot(steps, lr, color=RED, lw=1.7)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 1e5:.1f}e-5"))
ax.annotate(
    f"peak {lr.max():.2e} -> floor {lr[-1]:.2e} (10% cosine)",
    xy=(1.0, 0.06),
    xycoords="axes fraction",
    ha="right",
    color=DIM,
    fontsize=9,
)

fig.text(
    0.045,
    0.028,
    "aux_loss omitted: dense model (use_moe=0) has no MoE experts, so model_minimind.py:236 "
    "returns a zero tensor and logits_loss is identical to loss.",
    color=DIM,
    fontsize=9,
)

fig.text(
    0.045,
    0.955,
    "MiniMind dense 64M  |  SFT sft_t2t_mini  |  "
    f"{total:,} micro-steps  |  final loss {loss[-1]:.3f}",
    color=FG,
    fontsize=12,
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=160, facecolor=BG)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
print("parsed", len(steps), "log lines, iters", total)
