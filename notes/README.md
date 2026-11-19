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
├── lora.md                   # 训练：LoRA 冻结 base，只学低秩增量
├── ppo.md                    # 训练：MiniMind 的 PPO 与经典 PPO 的 8 处差异
└── assets/                   # 笔记配图
```
