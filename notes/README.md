# notes

学习笔记。按主题分目录，**一个文件只讲一件事**：代码在仓库里能读，笔记只记「为什么这么设计」。

```text
notes/
├── README.md                 # 本索引
├── dataflow.md               # 模型结构：数据怎么流、张量形状怎么变
├── params.md                 # 模型结构：参数怎么数
├── moe.md                    # 模型结构：MoE 稀疏激活（router 选专家）
├── pretrain.md               # 训练：一次参数更新里发生了什么
├── sft.md                    # 训练：SFT 只让回答段算 loss
└── assets/                   # 笔记配图
```

## 索引

| 主题 | 文件 | 一句话 |
|---|---|---|
| 数据流 | [dataflow.md](dataflow.md) | 一条输入从 `hello` 到 `world`，张量形状怎么变；`attn1` / `ffn1` 里各有什么 |
| 参数账 | [params.md](params.md) | 参数 = 表里的格子 = 行 × 列；64M 是 6400×768 + 8×一层 + 768 加出来的 |
| MoE | [moe.md](moe.md) | 稀疏激活：router 用 softmax + top-k 选专家；参数 $\times N$、每 token 算力不变 |
| 预训练 | [pretrain.md](pretrain.md) | 128 条样本换一个平均梯度；loss 8.76 的来历与 shift；Adam 对梯度缩放免疫 |
| 监督微调 | [sft.md](sft.md) | 对话套模板；同一个 CE，靠 labels 的 -100 只在回答段算 loss；从「会接龙」到「会回答」 |
