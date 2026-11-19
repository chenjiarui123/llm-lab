# MiniMind 的 PPO：和经典 RLHF-PPO 的差异对照


## 1. 写法对照（公式版）

| 主题 | 经典 | MiniMind |
|---|---|---|
| Critic 初始化 | $\phi \leftarrow \phi_{RM}$ | 主干 $\leftarrow$ `full_sft`，`value_head` 随机初始化（默认 `nn.Linear`） |
| KL 位置 | $r_t^{\text{classic}} = -\beta\log\frac{\pi_\theta}{\pi_{ref}}$ 进 reward | $r_t = \mathbb{1}[t=T]\cdot R$（只有末尾有分），KL 进 loss |
| KL 估计 | k1 | 惩罚用 $\text{k3}=e^{x}-x-1\ (\ge0)$；早停诊断另用 $\text{k2}=\tfrac12(\log r)^2$ |
| Reward | $r_\phi(q,o)$ | 规则分 + RM（**仅 RM 分 clip 到 $\pm3$**，规则分另加、总分不裁），RLAIF |
| Critic loss | $\tfrac12 (R-V)^2$ | $\tfrac12\max\!\big[(V-R)^2,\ (\mathrm{clip}(V,\,V_{old}\pm0.2)-R)^2\big]$ |
| 优势 | $\hat A_t$ | $\tilde A_t = \dfrac{\hat A_t - \mu_A}{\sqrt{\sigma_A^2+\epsilon}}\,m_t$ |
| 更新 | 固定 $K=4$ | $K=2$，且 $\widehat{\mathrm{KL}}_{k2} > 0.25$ 早停 |
| 采样 | 一题一条 | $\text{num\_generations}=1,\ T=0.8$（与 GRPO 的分界） |

## 2. 为什么这么改

| 主题 | 为什么 | 影响 |
|---|---|---|
| Critic 初始化 | 不单训 critic：直接复用 `full_sft` 主干（省掉「训自己的 RM、再从 RM 初始化 critic」）；RM 用现成的 InternLM2-1.8B | 早期 $V$ 不准，收敛慢一点 |
| KL 位置 | 解耦、可单独调；再另算 `approx_kl` 做早停 | reward 变稀疏（只有末尾 $R$）；critic 只拟合终局回报，学不到 KL |
| KL 估计 k3 | 小样本更稳 | 恒非负、低方差，**KL 惩罚项**更稳 |
| Reward 规则+RM | 小 RM 不稳，先用规则兜住格式/复读 | 更抗 reward hacking；代价是引入长度/格式先验 |
| Critic loss clip | 与 actor 的 clip 对称 | 防 critic 一步跳飞 |
| 优势 masked 白化 | pad 不该进统计量 | 梯度尺度稳，pad 不污染 |
| 2 轮 + 早停 | 自适应，省时 | 跑偏立刻停（早停看 $\text{k2}=\tfrac12(\log r)^2$，阈值 0.25） |
| 单条采样 | 本实现 1 题 1 条（on-policy） | 与 GRPO（一题多采、组内均值当基准）的分界 |

## 3. 一次训练循环

一批 prompt 走一遍：

```
prompt → ①rollout(actor 采样, 记 π_old) → ②打分 R(规则+RM, 只在末 token)
      → ③critic 估 V / ref 算 log π_ref → ④GAE + 优势白化
      → ⑤K=2 轮更新(π_θ 与 V) → ⑥同步权重 → 下一批
```
## 4. 两个 loss

重要性比 $r_t(\theta)=\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}=e^{\log\pi_\theta-\log\pi_{old}}$，$\tilde A_t$ 为白化后的优势，二者都按回答段 mask 取平均。

**① Actor / policy loss（clip + KL）**
$$
\mathcal L_{\text{policy}}=-\,\mathbb E_t\!\Big[\min\!\big(r_t\tilde A_t,\ \mathrm{clip}(r_t,\,1-\epsilon,\,1+\epsilon)\,\tilde A_t\big)\Big]\;+\;c_2\,\mathbb E_t\!\big[k3_t\big]
$$
其中 $k3_t=e^{d}-d-1,\ d=\log\pi_{ref}-\log\pi_\theta\ (\ge 0)$，$\epsilon=0.2$，$c_2=0.02$。（代码写成 $\max(-\tilde A\,r,\,-\tilde A\,\mathrm{clip}(r))$，与 $-(\cdot)$ 前一项等价。）
前一项是 ratio + clip 的信任域，限制这一步能改多少；后一项是 KL 惩罚（k3），把 $\pi_\theta$ 拽住别离 $\pi_{ref}$ 太远。

**② Critic / value loss（带 clip 的 MSE）**
$$
\mathcal L_{\text{value}}=\frac12\,\mathbb E_t\!\Big[\max\!\Big((V_t-R_t^{\text{ret}})^2,\ \big(\mathrm{clip}(V_t,\,V_t^{old}\pm\epsilon_v)-R_t^{\text{ret}}\big)^2\Big)\Big]
$$
$R_t^{\text{ret}}$ 即 `returns`，$\epsilon_v=$ `cliprange_value` $=0.2$。

**③ 总 loss（每个 mini-batch）**
$$
\mathcal L=\mathcal L_{\text{policy}}+c_1\,\mathcal L_{\text{value}}+\text{aux\_loss}
$$
$c_1=$ `vf_coef` $=0.5$；`aux_loss` 只有 MoE 才非 0（dense 为 0）；再除以梯度累积步数后 backward。早停时把 loss 乘 0 只为保持 DDP 通信闭环，不再更新。
