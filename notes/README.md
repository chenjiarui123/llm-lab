# notes

学习笔记。按主题分目录，**一个文件只讲一件事**：代码在仓库里能读，笔记只记「为什么这么设计」。

```text
notes/
├── README.md                 # 本索引
├── dataflow.md               # 模型结构：数据怎么流、张量形状怎么变
├── params.md                 # 模型结构：参数怎么数
├── pretrain.md               # 训练：一次参数更新里发生了什么
└── assets/                   # 笔记配图
```

## 索引

| 主题 | 文件 | 一句话 |
|---|---|---|
| 数据流 | [dataflow.md](dataflow.md) | 一条输入从 `hello` 到 `world`，张量形状怎么变；`attn1` / `ffn1` 里各有什么 |
| 参数账 | [params.md](params.md) | 参数 = 表里的格子 = 行 × 列；64M 是 6400×768 + 8×一层 + 768 加出来的 |
| 预训练 | [pretrain.md](pretrain.md) | 128 条样本换一个平均梯度；loss 8.76 的来历与 shift；Adam 对梯度缩放免疫 |
