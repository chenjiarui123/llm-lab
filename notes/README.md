# notes

学习笔记。按主题分目录，**一个文件只讲一件事**：代码在仓库里能读，笔记只记「为什么这么设计」。

```text
notes/
├── README.md                 # 本索引
├── architecture/             # 模型结构：数据怎么流、参数怎么数
│   ├── dataflow.md
│   └── params.md
└── assets/                   # 笔记配图
```

## 索引

| 主题 | 文件 | 一句话 |
|---|---|---|
| 数据流 | [architecture/dataflow.md](architecture/dataflow.md) | 一条输入从 `hello` 到 `world`，张量形状怎么变；`attn1` / `ffn1` 里各有什么 |
| 参数账 | [architecture/params.md](architecture/params.md) | 参数 = 表里的格子 = 行 × 列；64M 是 6400×768 + 8×一层 + 768 加出来的 |

## 待写

- [ ] tokenizer：BPE 的合并规则怎么从语料里长出来
- [ ] pretrain 与 SFT：数据格式和 loss 的差别
- [ ] LoRA：低秩分解到底省在哪
- [ ] KV cache：为什么 GQA 能省显存
