# llm-lab

从零学大模型的实验工作区。本地（Windows / 8GB 显存）与远端（AutoDL）共用同一份代码，
只有配置不同；**代码走 Git，数据和权重走 ModelScope / Transmit，环境每台机器各自重建**。

```
llm-lab/
├── README.md
├── .gitignore
├── requirements/        # 依赖拆分
├── scripts/             # 环境/下载/训练/评测 入口
├── configs/             # 机器相关参数（唯一需要改的地方）
├── modern-gpt/          # 自己实现的现代 GPT（对照学习用）
├── experiments/         # 每次实验的配置、指标、结论
├── dataset/             # [ignored] jsonl 数据
├── out/                 # [ignored] 训练权重 .pth
└── third_party/minimind # [ignored] 用脚本 clone 的 MiniMind 源码
```

---

## 0. 一句话流程

```text
git clone llm-lab → setup 环境 → download 数据 → train_pretrain → train_sft → eval
```

---

## 1. 环境

### Windows（本地 8GB）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

### Linux / AutoDL

```bash
bash scripts/setup_autodl.sh
```

两者都会做同一件事：

1. 建 `.venv`（Windows 用 `--system-site-packages` 复用已装好的 torch）
2. 装 `requirements/base.txt` + `requirements/train.txt`
3. `git clone` MiniMind 到 `third_party/minimind` 并打印 commit

> `third_party/` 不进 Git。换机器时重新 clone，保证两边代码完全一致。

---

## 2. 数据

只下载 mini 版，两个文件约 2.8GB：

| 文件 | 大小 | 用途 |
|---|---|---|
| `pretrain_t2t_mini.jsonl` | ~1.2GB | 预训练 |
| `sft_t2t_mini.jsonl` | ~1.6GB | 指令微调 |

```bash
python scripts/download_data.py --mini
```

数据源（作者托管，二选一）：

- ModelScope：<https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files>
- HuggingFace：<https://huggingface.co/datasets/jingyaogong/minimind_dataset>

下完会落在 `dataset/`。**不要下 `pretrain_t2t.jsonl` / `sft_t2t.jsonl`（10GB / 14GB 全量版）。**

已有数据文件想跳过下载：

```bash
python scripts/download_data.py --check
```

---

## 3. 训练

选配置 → 跑脚本。机器差异全部在 `configs/` 里，脚本本身不改。

```bash
# 本地 8GB
bash scripts/train_pretrain.sh configs/local_8gb.yaml
bash scripts/train_sft.sh      configs/local_8gb.yaml

# AutoDL 24GB
bash scripts/train_pretrain.sh configs/autodl_24gb.yaml
bash scripts/train_sft.sh      configs/autodl_24gb.yaml
```

Windows 没有 bash 时，直接用 Python 启动器（等价）：

```powershell
python scripts/launch.py pretrain --config configs/local_8gb.yaml
python scripts/launch.py sft      --config configs/local_8gb.yaml
```

只想验证能跑通（不保存正式权重）：

```powershell
python scripts/launch.py pretrain --config configs/local_8gb.yaml --extra --save_interval 999999
```

### 产出

```text
out/pretrain_768.pth      # 预训练权重
out/full_sft_768.pth      # SFT 后的对话权重
```

---

## 4. 评测

```bash
bash scripts/eval.sh full_sft
bash scripts/eval.sh pretrain
```

等价于：

```bash
python third_party/minimind/eval_llm.py \
  --load_from third_party/minimind/model \
  --save_dir out --weight full_sft
```

---

## 5. 8GB 显存怎么撑住

\[
\text{显存} \approx \text{batch\_size} \times \text{max\_seq\_len} \times \text{常数}
\]

OOM 时按这个顺序降，**不要动模型结构**：

```text
batch_size:      8 → 4 → 2
max_seq_len:   512 → 384 → 256 → 192
num_workers:     8 → 4 → 2
```

`accumulation_steps` 反过来加倍，保持有效 batch 不变：

\[
\text{有效 batch} = \text{batch\_size} \times \text{accumulation\_steps}
\]

所以 `8×16 = 4×32 = 2×64`。

---

## 6. 判断“训练是对的”

Pretrain 初始 loss 约：

\[
\ln(6400) \approx 8.76
\]

只要 loss 从 8.x 往下走，说明数据、tokenizer、前向反向都通了。

---

## 7. 实验记录

每次实验在 `experiments/` 下建目录：

```text
experiments/2026-11-07-dense64m/
├── config.yaml    # 当次实际用的配置（拷贝一份，不要引用）
├── metrics.csv    # step,loss,lr,备注
└── README.md      # 假设 / 结果 / 结论 / 踩坑
```

只提交这三个小文件，不提交权重和日志。

---

## 8. 三台机器怎么同步

```text
代码：  Mac ↔ GitHub ↔ Windows / AutoDL     （git）
数据：  ModelScope → 各机器下载             （脚本 / Transmit）
权重：  AutoDL → Mac 备份 → Windows 按需    （Transmit / 云盘）
环境：  每台机器各自重建                    （setup 脚本）
```

`.gitignore` 已经把 `dataset/` `out/` `third_party/` `.venv/` 全部排除。

---

## 9. 和 modern-gpt 的关系

| 目录 | 角色 |
|---|---|
| `modern-gpt/` | 自己写的现代 GPT，用来“理解每个部件” |
| `third_party/minimind/` | 成熟参考实现，用来“跑完整流程” |

推荐用法：先在 `modern-gpt` 里改一个模块（RMSNorm / RoPE / GQA / SwiGLU），
再回 MiniMind 看同部件在真实工程里怎么落地。
