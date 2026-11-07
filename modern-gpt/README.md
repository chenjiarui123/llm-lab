# modern-gpt：一个模块一个模块换

这个目录是给你做 **nanoGPT → 现代架构** 对比实验用的。

> 说明：`ng-video-lecture` 是 Karpathy 的原版教学源码；`modern-gpt` 是在它的结构基础上重写的中文注释实验版，不是直接修改 `ng-video-lecture` 目录。

不要一次把 MiniMind 全抄过来。这里把每个现代组件都做成一个开关：

| 开关 | 作用 | 对应现代架构 |
|---|---|---|
| `--use-rmsnorm` | LayerNorm → RMSNorm | Llama / MiniMind |
| `--use-rope` | 学习式位置编码 → RoPE | Llama / MiniMind |
| `--use-swiglu` | GELU FFN → SwiGLU | Llama / MiniMind |
| `--n-kv-head 2` | MHA → GQA | Llama / MiniMind |
| `--use-cache` | 推理时使用 KV Cache | 推理加速 |
| `--tie-weights` | 输入 embedding 与 lm_head 共享权重 | GPT-2 风格 |

默认全部关闭，所以它就是一个 **nanoGPT 风格的 baseline**。

---

## 文件

```text
modern-gpt/
├── model.py    # 所有模型组件，每个现代部件都有开关
├── train.py    # 训练脚本，支持命令行开关
├── tests.py    # shape / RoPE / GQA / KV Cache 正确性测试
└── README.md
```

---

## 1. 先跑 baseline

在 `modern-gpt/` 目录下：

```bash
python train.py --max-iters 300
```

如果显存不够或想快速试：

```bash
python train.py --tiny --max-iters 100
```

你应该看到类似：

```text
switches: rmsnorm=False, rope=False, swiglu=False, kv_heads=6
parameters: ...
step 0 | train ... | val ...
```

记下：

```text
参数量
train loss
val loss
```

---

## 2. 一次只开一个开关

### 第 1 步：RMSNorm

```bash
python train.py --max-iters 300 --use-rmsnorm
```

对比：

- 参数量少了多少？
- loss 是否正常？
- 训练是否更稳定？

### 第 2 步：SwiGLU

```bash
python train.py --max-iters 300 --use-rmsnorm --use-swiglu
```

看 `MLP` 里：

```python
gate_proj
up_proj
down_proj
```

以及：

```python
F.silu(gate_proj(x)) * up_proj(x)
```

### 第 3 步：RoPE

```bash
python train.py --max-iters 300 --use-rmsnorm --use-swiglu --use-rope
```

注意：开启 RoPE 后，模型里不再有：

```python
position_embedding
```

位置信息由 `RotaryEmbedding` 注入到 Q、K。

### 第 4 步：GQA

```bash
python train.py --max-iters 300 --use-rmsnorm --use-swiglu --use-rope --n-kv-head 2
```

默认 `n_head=6`，所以 `--n-kv-head 2` 表示：

```text
6 个 query heads 共享 2 组 KV heads
```

每个 KV head 被 3 个 query heads 使用。

### 第 5 步：KV Cache

KV Cache 主要影响推理，不影响训练 loss。

先跑测试：

```bash
python tests.py
```

重点看：

```text
ok: KV cache equivalence
```

这表示：

```python
logits(use_cache=False) == logits(use_cache=True)
```

然后可以自己在 Python 里计时：

```python
import torch
from model import GPTConfig, ModernGPT

cfg = GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4,
                n_embd=128, dropout=0.0, use_rmsnorm=True,
                use_swiglu=True, use_rope=True, n_kv_head=2)
model = ModernGPT(cfg).eval()
idx = torch.randint(0, 65, (1, 1))

# 比较 generate 的耗时
```

---

## 3. 建议的实验记录

每跑一次，记一行：

| 版本 | 参数量 | train loss | val loss | tokens/s |
|---|---:|---:|---:|---:|
| baseline | | | | |
| + RMSNorm | | | | |
| + SwiGLU | | | | |
| + RoPE | | | | |
| + GQA (n_kv_head=2) | | | | |
| + KV Cache | | | | |

注意：**tiny model 上 loss 不一定更好**。小实验主要验证：

- shape 是否正确；
- 代码是否跑通；
- 参数量是否符合预期；
- 推理是否更快、显存是否更低。

---

## 4. 代码阅读顺序

### `model.py`

1. `RMSNorm`：看它和 `nn.LayerNorm` 的公式差别；
2. `RotaryEmbedding` / `apply_rotary_emb`：看 Q、K 怎么旋转；
3. `repeat_kv`：看 GQA 的 KV 怎么复制；
4. `CausalSelfAttention`：看 QKV 投影、mask、cache；
5. `MLP`：看 SwiGLU 的三个投影；
6. `Block`：看 Pre-LN 残差结构；
7. `ModernGPT`：看 embedding、blocks、lm_head。

### `tests.py`

先跑：

```bash
python tests.py
```

如果某个测试失败，先修 shape，不要继续训练。

---

## 5. 和 MiniMind 对照时看什么

打开 MiniMind 的模型文件（常见为 `model/model_minimind.py`），只找这些：

```text
RMSNorm
RotaryEmbedding
Attention
MLP
Block
MiniMindForCausalLM
```

对照表：

| 本目录 | MiniMind |
|---|---|
| `RMSNorm` | `RMSNorm` |
| `RotaryEmbedding` | `RotaryEmbedding` / `precompute_freqs_cis` |
| `CausalSelfAttention` | `Attention` |
| `MLP` | `MLP` / `FeedForward` |
| `Block` | `MiniMindBlock` |
| `ModernGPT` | `MiniMindForCausalLM` |

重点看配置字段：

```text
hidden_size
num_hidden_layers
num_attention_heads
num_key_value_heads
intermediate_size
max_position_embeddings
rope_theta
```

不要从 `train_pretrain.py` 开始读。先把模型文件看懂。

---

## 6. 验收标准

每完成一个模块，问自己：

1. 不看代码，能不能写出这个模块？
2. 能不能解释它解决了什么问题？
3. 能不能说出它和 nanoGPT 原版的区别？
4. 能不能通过 `tests.py` 的对应测试？

四个都过了，再进入下一个模块。

---

## 7. 推荐顺序

```text
baseline
  → RMSNorm
  → SwiGLU
  → RoPE
  → GQA
  → KV Cache
  → 读 MiniMind 模型文件
```

不要先学 MoE。把 dense Transformer 的这几个替换件吃透，再看 MoE 会轻松很多。
