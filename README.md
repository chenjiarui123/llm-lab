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
├── configs/                # 机器相关参数（唯一需要改的地方）
│   ├── autodl_24gb.yaml
│   └── local_8gb.yaml
├── dataset/                # [ignored] jsonl 数据
├── experiments/            # 每次实验的配置、指标、结论
│   └── 2026-11-07-dense64m/
├── nano-gpt/               # 教学模型目录
│   ├── model1.py           # 基线：nanoGPT 版（含训练循环，字符级）
│   └── model2.py           # 现代组件版（RMSNorm/RoPE/GQA/SwiGLU/KV Cache）
├── notes/                  # 学习笔记（按主题分目录）
├── out/                    # [ignored] 训练权重 .pth
├── requirements/           # 依赖拆分
│   ├── base.txt
│   └── train.txt
├── scripts/                # 环境/下载/训练/评测 入口
└── third_party/minimind/   # [ignored] 用脚本 clone 的 MiniMind 源码
```

> 笔记在 [notes/](notes/README.md)：结构、参数账、训练要点，按主题分目录。




## 0. 一句话流程


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



---

## 4. 实验记录

每次实验在 `experiments/` 下建目录：

```text
experiments/2026-11-07-dense64m/
├── config.yaml    # 当次实际用的配置（拷贝一份，不要引用）
├── metrics.csv    # step,loss,lr,备注
└── README.md      # 假设 / 结果 / 结论 / 踩坑
```

只提交这三个小文件，不提交权重和日志。

---

## 5. 三台机器怎么同步

```text
代码：  Mac ↔ GitHub ↔ Windows / AutoDL     （git）
数据：  ModelScope → 各机器下载             （脚本 / Transmit）
权重：  AutoDL → Mac 备份 → Windows 按需    （Transmit / 云盘）
环境：  每台机器各自重建                    （setup 脚本）
```

