# 2026-11-14 SFT（dense 64M，全量，AutoDL）

第一次在 AutoDL（单卡 **RTX 5090 32GB**）上跑通 MiniMind dense 64M 的**全量 SFT**：
从 `out/pretrain_768.pth` 出发，用全量 `sft_t2t_mini.jsonl` 做 1 epoch 指令微调，产出可对话的 `out/full_sft_768.pth`。

> 这个文件夹是**手动**建的实验记录（config / metrics / loss 图 / 结论 / 复现脚本），训练脚本本身不会自动生成——pretrain 那个 `2026-11-10-pretrain-full` 同理，也是手写的。

## 配置

- 硬件：AutoDL 单卡 **RTX 5090 32GB**，bf16（训练显存占用约 7.9GB，GPU 利用率 93–98%）
- 模型：hidden 768 / 8 层 / `use_moe: 0`，63,912,192 参数（与 pretrain 同一个 dense 64M）
- 训练：`batch_size 16 × accumulation_steps 2` = **有效 batch 32**，`max_seq_len 768`，lr 1e-5 余弦退火到 1e-6（10% 下限），1 epoch
- 数据：`dataset/sft_t2t_mini.jsonl`，**905,718 条**多轮对话 → **56,608 micro-step ≈ 28,304 次优化器更新**
- 起点：`out/pretrain_768.pth`（`from_weight: pretrain`）

## 结果

| 指标 | 数值 |
|---|---|
| 起止 | 2026-11-14 10:54 → 11:58 |
| 耗时 | **约 1 小时 4 分**（≈0.067 s/step） |
| 退出码 | 0 |
| loss 起点 | 前 1000 步均值 2.114 |
| loss 终点 | 最后一步 **1.701**；末 20 个记录点均值 1.774 |
| 全程最低单步 | 1.334（约 52k 步处） |
| lr 终点 | 1.00e-6（余弦下限 10%） |
| 权重 | `out/full_sft_768.pth`，137,684,407 B（fp16，91 个张量） |

### loss 里程碑

| step | 进度 | loss | lr |
|---:|---:|---:|---:|
| ≤1,000（均值） | 1.8% | 2.114 | 1.00e-5 |
| 5,000 | 8.8% | 1.898 | 9.83e-6 |
| 10,000 | 17.7% | 1.885 | 9.32e-6 |
| 15,000 | 26.5% | 1.680 | 8.53e-6 |
| 20,000 | 35.3% | 1.739 | 7.50e-6 |
| 30,000 | 53.0% | 1.798 | 5.08e-6 |
| 40,000 | 70.7% | 1.812 | 2.78e-6 |
| 50,000 | 88.3% | 1.697 | 1.30e-6 |
| **56,608** | 100% | **1.701** | 1.00e-6 |

## 结论

1. **前 ~15k 步（约 26%）就吃掉了主要降幅**（2.37 → 1.68），之后 4 万步只在 1.70–1.81 之间平台震荡；**1 epoch 已基本收敛**，再加 epoch 收益很小。
2. 单步波动很大（1.33–2.37，取决于该 batch 回答长度/难度），**看滑动平均（EMA）才有意义**；蓝线 raw、红线 EMA。
3. `aux_loss` 全程恒为 0：dense 模型没有 MoE 专家，`model_minimind.py:236` 对空列表求和返回零张量，`logits_loss == loss`。
4. **SFT loss 只在「回答段」上算**（其余位置 `label = -100`），它衡量的是「模仿回答」的拟合程度，**不等于对话质量**。要判断效果得跑 `eval_llm.py` 实际生成（见下）。
5. 和 pretrain 的 1.80 不能直接比：pretrain 是**全序列**续写 loss，SFT 是**仅回答段** loss，口径不同。

## 复现

```bash
cd ~/llm-lab
source .venv/bin/activate
bash scripts/train_sft.sh experiments/2026-11-14-sft-full/config.yaml > experiments/2026-11-14-sft-full/train.log 2>&1
```

`run.sh` 就是上面这段的封装（带 `PYTHONUNBUFFERED=1`，日志进 `train.log`）。

曲线由 `plot.py` 从 `train.log` 解析生成：

```bash
python experiments/2026-11-14-sft-full/plot.py experiments/2026-11-14-sft-full/train.log experiments/2026-11-14-sft-full/loss.png
```

推理看效果：

```bash
bash scripts/eval.sh full_sft
```

注意 `train.log` 与 `*.pth` 都被 `.gitignore` 排除，仓库里只留 `loss.png` 这张成品图。
