# 2026-11-10 pretrain（dense 64M，全量）

第一次全量跑通 MiniMind dense 64M 的 pretrain：验证链路，产出一个能进 SFT 的 base 权重。

## 配置

- 硬件：RTX 5060 Ti 8GB，bf16
- 模型：hidden 768 / 8 层 / 8 Q 头 / 4 KV 头 / 词表 6400 / FFN 2432，`use_moe: 0`，63,912,192 参数
- 训练：batch 8 × accum 16 = 有效 128，`max_seq_len` 256，lr 5e-4 余弦退火到 5e-5，1 epoch
- 数据：`dataset/pretrain_t2t_mini.jsonl`，1,270,240 条 → **158,780 micro-step ≈ 9,924 次优化器更新**

## 结果

| 指标 | 数值 |
|---|---|
| 起止 | 2026-11-10 07:25:38 → 09:15:17 |
| 耗时 | **1 小时 49 分 39 秒**（0.045 s/step） |
| 退出码 | 0 |
| loss 起点 | 前 1000 步均值 7.05 |
| loss 终点 | 最后一步 **1.802**；末 9000 步均值 2.00 |
| lr 终点 | 5.00e-5（余弦下限 10%） |
| 权重 | `out/pretrain_768.pth`，137,684,407 B |

### loss 里程碑

| step | 进度 | loss | lr |
|---:|---:|---:|---:|
| ≤1,000（均值） | 0.6% | 7.05 | 5.00e-4 |
| 10,000 | 6.3% | 3.60 | 4.96e-4 |
| 20,000 | 12.6% | 2.88 | 4.83e-4 |
| 40,000 | 25.2% | 2.22 | 4.33e-4 |
| 80,000 | 50.4% | 2.60 | 2.72e-4 |
| 120,000 | 75.6% | 2.05 | 1.13e-4 |
| 150,000 | 94.5% | 2.26 | 5.34e-5 |
| **158,780** | 100% | **1.80** | 5.00e-5 |

## 结论

1. **前 20k 步（12.6% 的算力）吃掉了 6.2 个 nat 里的 5.2 个**；后 87% 只赚 1 个 nat。64M 容量 + 1.27M 条数据，尾部早就平了，加 step 的边际收益极低。
2. loss ≈ 2.0 → 困惑度 $e^{2.0}\approx 7.4$，即每个 token 在 7.4 个候选里犹豫。这是一个**掌握了语法和词频的续写器**，不是会听指令的模型。
3. `aux_loss` 全程恒为 0：dense 模型没有 MoE 专家，`third_party/minimind/model/model_minimind.py:236` 对空列表求和返回零张量，所以日志里 `logits_loss` 与 `loss` 完全相同。

## 复现

```bat
cd /d E:\3_llm\llm-lab
experiments\2026-11-10-pretrain-full\run.bat  :: 脱离会话后台跑，日志进 train.log
python scripts\launch.py pretrain --config experiments\2026-11-10-pretrain-full\config.yaml
```

曲线由 `plot.py` 从 `train.log` 解析生成：

```bat
E:\ai\miniconda3\envs\torch\python.exe experiments\2026-11-10-pretrain-full\plot.py
```

注意 `train.log` 与 `*.pth` 都被 `.gitignore` 排除，仓库里只留 `loss.png` 这张成品图。
