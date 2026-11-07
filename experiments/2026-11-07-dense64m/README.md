# 2026-11-07 dense 64M 全流程

## 假设 / 目标

第一次完整走通四步，验证脚本链路而不是追求指标：

```text
数据下载 → pretrain → sft → eval
```

判据只有三条：

1. pretrain loss 从约 `ln(6400) ≈ 8.76` 明显下降
2. 8GB 显存不 OOM
3. `full_sft_768.pth` 能生成连贯中文

## 配置

见同目录 `config.yaml`（实际执行的快照）。

| 项 | 值 |
|---|---|
| 模型 | dense，hidden 768，8 层，非 MoE（约 64M 参数） |
| 机器 | 本地 RTX 5060 Ti 8GB |
| pretrain | batch 8 × accum 16 = 有效 128，seq 256 |
| sft | batch 4 × accum 4 = 有效 16，seq 512 |
| dtype | bfloat16 |

## 执行的命令

```bash
python scripts/download_data.py --mini
bash scripts/train_pretrain.sh configs/local_8gb.yaml
bash scripts/train_sft.sh      configs/local_8gb.yaml
bash scripts/eval.sh full_sft
```

## 结果

| stage | 起始 loss | 结束 loss | 耗时 | 显存峰值 |
|---|---|---|---|---|
| pretrain | | | | |
| sft | | | | |

填完把关键行同步到 `metrics.csv`，日志不要提交。

## 结论

（跑完再写：loss 是否正常下降、sft 后对话是否像样、哪一步最慢）

## 踩坑

（OOM 出现在第几步、降了什么参数才过、数据路径/编码问题等）

---

## 记录模板（下次复制这个目录）

```text
1. mkdir experiments/YYYY-MM-DD-<主题>
2. cp configs/local_8gb.yaml experiments/YYYY-MM-DD-<主题>/config.yaml
3. 只改这一份副本，改完再用于实际训练
4. 填 metrics.csv + README
```
