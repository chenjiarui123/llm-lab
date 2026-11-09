# 预训练：一次参数更新里到底发生了什么

以 MiniMind dense 64M（hidden 768、8 层、词表 6400）、`batch_size=8`、`accumulation_steps=16`、`max_seq_len=256`、bf16 为例。

## 1. 四个量，别混

| 名字 | 含义 | 你的配置 |
|---|---|---|
| epoch | 把整个数据集过一遍 | 1 轮 |
| batch（microbatch） | 一次前向吃多少条 | 8 条 |
| iters | 一轮有多少个 batch | $\lceil N/8\rceil$ |
| step | 第几个 batch（从 1 数到 iters） | 1 … iters |

$$\text{iters}=\left\lceil\frac{N}{B}\right\rceil,\qquad \text{更新次数}=\left\lceil\frac{\text{iters}}{A}\right\rceil,\qquad \text{总更新}=\text{epochs}\times\left\lceil\frac{\text{iters}}{A}\right\rceil$$

**step ≠ 参数更新次数。** 每 $A=16$ 个 step 才更新一次参数，所以 `step 500` 实际只更新了 31 次。

## 2. 一次更新 = 六步，顺序不能换

```text
16 个 microbatch（各 8 条）            = 有效 batch 128
  └─ 每个：前向 → loss/16 → backward（梯度累加）
       ↓
  128 条样本的平均梯度
       ↓
  unscale → clip_grad_norm(1.0) → AdamW step → zero_grad
```

| 步骤 | 干什么 | 换个顺序会怎样 |
|---|---|---|
| `backward()` | 链式法则求 $\partial L/\partial W$，存进 `W.grad`（**累加**，不是覆盖） | — |
| `loss / A` | 让 16 次累加变成平均，否则梯度大 16 倍 | 见 §5 的 Adam 陷阱 |
| `unscale_` | fp16 的梯度被放大过，裁剪前先还原 | 裁的是假数，等于没裁 |
| `clip_grad_norm_(1.0)` | $\lVert g\rVert>1$ 时整体缩回，只改长度不改方向 | 裁剪后覆盖梯度就白裁了 |
| `step()` | $\theta \leftarrow \theta-\eta\cdot\hat{g}$（AdamW：动量 + 自适应步长） | 清了梯度就没得更新 |
| `zero_grad` | 梯度置 `None`，为下轮累加清场 | 不清就累加成上一轮的 2 倍 |

$\hat{g}=\hat{m}/\sqrt{\hat{v}}$ 里每个参数大致落在 $[-1,1]$，所以 $\Delta\theta\approx\eta\approx5\times10^{-4}$：**步长由 $\eta$ 统一管，与梯度大小基本无关**，这是 Adam 和 SGD 的根本区别。

## 3. 学习率曲线：开跑前就能算出来

$$\eta(t)=\eta_{\max}\left(0.1+0.45\left(1+\cos\frac{\pi t}{T}\right)\right)$$

| 符号 | 含义 | 代码 |
|---|---|---|
| $t$ | **全局**步号 | `epoch * iters + step` |
| $T$ | 总步数 | `epochs * iters` |

| $t/T$ | 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| 倍率 | **1.000** | 0.868 | 0.550 | 0.232 | **0.100** |

- **除以 $T$ 只是把横轴刻度换成「进度百分比」**：$\cos$ 需要 $0\sim\pi$ 的角度，不管训练 1 万步还是 100 万步，形状都是「满速起步、10% 收尾」。
- $t$ 必须用全局步号。写成局部的 `step`，第二轮会满速重开，训练直接崩。
- **$B\times A$ 不变时曲线一分没动**：$B$ 8→4、$A$ 16→32，第 $k$ 次更新时的 $t/T$ 完全相同，只是日志步号翻倍。
- 想改曲线只有两条路：改 `epochs`（拉伸 $T$）或改 `learning_rate`（整体缩放）。

## 4. loss 为什么从 8.76 开始

$$\text{logits}\ [B,T,6400]\xrightarrow{\ \text{softmax}\ }p_i,\qquad \mathcal{L}=-\log p_c$$

随机初始化时输出几乎均匀，$p_c\approx1/6400$：

$$\mathcal{L}\approx-\log\frac{1}{6400}=\ln 6400=8.7641$$

这就是「完全瞎猜」的代价，**不是误差**。第一个 step 落在 8.7~8.9 说明接线全对。

### shift：位置 $t$ 的输出预测位置 $t+1$ 的词

| 下标 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| token | 1 (bos) | 5 | 9 | 13 | 21 | 2 (eos) |
| `x` 用谁的 logits | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| `y` 要预测谁 | 5 | 9 | 13 | 21 | 2 | — |

- 砍的是**最后一个位置的输出**和**第一个位置的标签**，与 token 是否特殊无关。
- **eos 必须当预测目标**，否则模型学不会「说完了停」，生成时会一直输出下去。
- shift 在模型里做（`logits[..., :-1, :]` 对 `labels[..., 1:]`），数据集里 `labels = input_ids.clone()`。**自己再 shift 一次 = 错一位，loss 卡在 8.7 降不下去**。
- `ignore_index=-100` 让 pad 位置既不算 loss 也不进分母。

| loss | 困惑度 $e^{\mathcal{L}}$ | 含义 |
|---|---|---|
| 8.76 | 6400 | 在全部词表里瞎猜 |
| 4.00 | 55 | 常见搭配开始对 |
| 2.50 | 12 | 句子通顺 |
| 1.00 | 2.7 | 基本在背诵 |

## 5. 梯度累积：用时间换显存

$$\text{显存}\approx\underbrace{\text{参数}+\text{梯度}+\text{优化器状态}}_{\text{与 }B\text{ 无关}}+\underbrace{\text{激活值}}_{\propto\,B\times T\times L}$$

64M 参数的固定部分约 1.02 GB（bf16 权重 2B + 梯度 2B + AdamW 的 m/v 8B + fp32 主权重 4B，合计 16 B/参数，$63.9\text{M}\times16\text{B}\approx1.02\text{GB}$），**大头是激活值，正比于 $B\times T$**。所以「有效 batch 128」能跑，靠的是 16 次小前向，而不是真塞 128 条。

- **和真·大 batch 等价**：LayerNorm 只在单样本内部归一化，不像 BN 那样依赖跨样本统计。
- **代价**：16 次串行、GPU 利用率低，比真·大 batch 慢。

### 一个反直觉的点

删掉 `loss / A`：梯度整体 ×16。但 **Adam 对这种缩放免疫**（$m$ 与 $\sqrt{v}$ 同时被放大，比值不变），所以不是「学习率 ×16 炸掉」，而是——

> 梯度范数恒定超过 `grad_clip=1.0` → **裁剪每步都触发 → 自适应步长被压成固定小步 → loss 降得越来越慢，而且不报错。**

（SGD 才是线性的：$g\to16g$ 严格等于 $\eta\to16\eta$。）

## 6. 混合精度：训练要范围，推理要精度

| 类型 | 位 | 指数 | 尾数 | 范围 | 相对精度 |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | $10^{\pm38}$ | $10^{-7}$ |
| fp16 | 16 | **5** | 10 | $6\times10^{-5}\sim65504$ | $5\times10^{-4}$ |
| bf16 | 16 | **8** | 7 | $10^{\pm38}$ | $4\times10^{-3}$ |

都是 16 位，钱花的地方不同：**bf16 买范围（同 fp32，不会上下溢出），fp16 买精度（多 3 位尾数，但范围窄）。**

`autocast` 是白名单，不是全转：

| 走 bf16 | 保持 fp32 |
|---|---|
| 矩阵乘、线性层、$QK^\top$、$AV$ | RMSNorm 的均值方差、softmax、**交叉熵**、参数更新 |

softmax / loss 必须 fp32：$e^{z}$ 对范围极敏感，$p$ 被冲成 0 就是 $\log 0=-\infty$ → `loss=nan`，一步污染全部参数。

`GradScaler(enabled=(dtype=='float16'))`：跑 bf16 时它**全程空转**（scale/unscale/step 全是直通）。fp16 需要它，是因为梯度常是 $10^{-7}$，低于 fp16 最小正规数 $6.1\times10^{-5}$ → **下溢成 0 → 参数悄悄不更新**（不是溢出；溢出反而好查）。

| 场景 | 选择 | 原因 |
|---|---|---|
| 训练 | bf16 | 不用 scaler，怕 nan |
| 推理 / 存盘 | fp16 | 不要梯度、不会下溢，多 3 位精度 |

## 7. 数据构造：不是 BPE，是「切完怎么拼」

```text
jsonl 一行 {"text": "..."}
  → tokenizer(..., max_length=T-2, truncation=True)   从右侧截断
  → [bos] + ids + [eos]                               凑到 T-1 ~ T
  → pad 到 T                                          补 pad_token_id
  → labels = input_ids，pad 位置改 -100
```

- **为什么是 `T-2`**：最终张量必须定长，才能 `stack` 成 $[B,T]$。先给 bos/eos 留两个位置，否则加完就超长。
- **pad 无害的三个条件**：因果掩码（位置 $i$ 只看 $\le i$）+ pad 在**末尾**（真实 token 看不见它）+ `labels=-100`（不产生梯度）。三条齐全，预训练才能省掉 `attention_mask`。
- **反例**：pad 放最前面（左 padding，推理批生成常用）会污染所有真实 token，那时必须传 `attention_mask`。

| `max_seq_len` | 约汉字数（1 token ≈ 1.5~1.7 字） | 8GB |
|---|---|---|
| 256 | 380~430 | 舒服 |
| 340（代码默认） | 510~580 | 偏紧 |
| 512 | 770~870 | 要 batch 2 |

600 字的样本在 256 下只保留约 2/3，**被砍掉的是句尾**。

## 8. 开跑前的排错表

| 症状 | 大概率原因 |
|---|---|
| 首步 loss ≠ 8.7~8.9 | vocab_size 不一致 / labels 自己 shift 过 / 加载了已训权重 |
| loss 完全不降 | 梯度被 clip 恒定截断（如忘了 `/A`）/ lr 太小 / 标签全 -100 |
| `loss=nan` | fp16 溢出、lr 太大、在 bf16 下算交叉熵 |
| loss 降得极慢且无报错 | fp16 梯度下溢（缺 GradScaler） |
| OOM | `batch_size` 8→4→2，再降 `max_seq_len`，`accumulation_steps` 反比加倍 |

## 一句话串起来

**128 条样本算出一个平均梯度，按余弦曲线给定的步长把它从参数里减掉；loss 从「全词表瞎猜」的 8.76 出发，靠「位置 $t$ 猜 $t+1$」这条唯一规则，把每个位置猜中的概率从 $1/6400$ 一点点抬起来。**
