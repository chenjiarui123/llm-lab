# llm-lab

从零学大模型的实验工作区。本地（Windows / 8GB 显存）与远端（AutoDL）共用同一份代码，
只有配置不同：**代码走 Git，数据和权重走 ModelScope / Transmit，环境每台机器各自重建**。

---

## 目录（A → Z）

```text
llm-lab/
├── .gitattributes          # 换行符策略：.sh/.py 强制 LF
├── .gitignore              # dataset/ out/ third_party/ 全部排除
├── README.md
├── archives/               # [ignored] 压缩包、临时大文件
├── configs/                # 机器相关参数（唯一需要改的地方）
│   ├── autodl_24gb.yaml
│   └── local_8gb.yaml
├── dataset/                # [ignored] jsonl 数据
├── experiments/            # 每次实验的配置、指标、结论
│   └── 2026-11-07-dense64m/
├── modern-gpt/             # 现代组件版 GPT（RMSNorm/RoPE/GQA/SwiGLU/KV Cache）
│   └── model.py
├── nanogpt/                # nanoGPT 版（教学用，含训练循环）
│   └── model.py
├── out/                    # [ignored] 训练权重 .pth
├── requirements/           # 依赖拆分
│   ├── base.txt
│   └── train.txt
├── scripts/                # 环境/下载/训练/评测 入口
└── third_party/minimind/   # [ignored] 用脚本 clone 的 MiniMind 源码
```

### 逐个说明

| 目录 | 作用 | 进 Git |
|---|---|---|
| `archives/` | 压缩包、临时大文件，随时可删 | 否 |
| `configs/` | 每台机器一套 yaml，脚本不改、只换配置 | 是 |
| `dataset/` | `pretrain_t2t_mini.jsonl`、`sft_t2t_mini.jsonl` | 否 |
| `experiments/` | 一次实验一个目录：`config.yaml` + `metrics.csv` + `README.md` | 是 |
| `modern-gpt/` | 自己写的现代 GPT，按模块替换学习 | 是 |
| `nanogpt/` | Karpathy 教学版，单文件、能直接跑 | 是 |
| `out/` | `pretrain_768.pth`、`full_sft_768.pth` | 否 |
| `requirements/` | `base.txt` 通用依赖，`train.txt` 训练依赖 | 是 |
| `scripts/` | setup / download / launch / train / eval | 是 |
| `third_party/minimind/` | 第三方源码，clone 得到，只记录 commit | 否 |

### 两个模型目录的区别

| | `nanogpt/model.py` | `modern-gpt/model.py` |
|---|---|---|
| 来源 | `ng-video-lecture/gpt.py`（带中文注释） | 自己按模块重写 |
| 形态 | 单文件，含训练循环，可独立运行 | 纯模块：只定义类，不含训练 |
| 组件 | LayerNorm、learned 位置编码、GELU、MHA | RMSNorm、RoPE、SwiGLU、GQA、KV Cache |
| 用途 | 当**骨架和基线** | 当**替换实验的答案册** |
| 参数量 | 约 10.8M | 约 9.5M（全 modern） |

推荐顺序：先读 `nanogpt/model.py` 看懂结构 → 再到 `modern-gpt/model.py` 一个模块一个模块替换。

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

只检查本地是否齐全：

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

只看命令不执行：

```powershell
python scripts/launch.py pretrain --config configs/local_8gb.yaml --dry-run
```

### 为什么要 `launch.py` 这一层

MiniMind 的 trainer 内部写死了两处相对路径：

```python
init_model(...)                        # tokenizer_path 默认 '../model'
lm_checkpoint(..., save_dir='../checkpoints')
```

所以必须在 `third_party/minimind/trainer/` 下运行，但数据和权重在 `llm-lab/` 下。
`launch.py` 负责：读 yaml → 拼成 `--key value` → 切目录 → 启动子进程。

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

GitHub 远端第一次推送：

```powershell
cd E:\3_llm\llm-lab
git remote add origin https://github.com/<用户名>/llm-lab.git
git push -u origin main
```

没有 GitHub CLI（`gh`）也能推送，`git` 自带凭据管理即可。
装了 `gh` 之后可以先 `gh auth login` 再自动带凭据，属于可选项。
