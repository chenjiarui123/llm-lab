"""一个用于学习现代架构的迷你 GPT-2 风格模型（中文注释版）。

每个现代组件都放在一个配置开关后面：

    use_rmsnorm  : LayerNorm -> RMSNorm
    use_rope     : 学习式位置编码 -> RoPE
    use_swiglu   : GELU FFN -> SwiGLU FFN
    n_kv_head    : MHA -> GQA（None 表示和 n_head 相同）
    use_cache    : 推理时使用 KV cache

目标不是做生产级实现，而是做一个实验台：
每次只打开一个开关，然后和 baseline 对比。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False

    # 现代架构开关
    use_rmsnorm: bool = False
    use_rope: bool = False
    use_swiglu: bool = False
    n_kv_head: int | None = None      # None -> 普通 MHA
    use_cache: bool = False           # 只被 generate() 使用
    tie_weights: bool = False
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """均方根归一化（不减均值、不使用 bias）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """预计算 RoPE 所需的 cos/sin 表。"""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE 要求 head_dim 为偶数"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"序列长度 {seq_len} 超过 RoPE cache {self.max_seq_len}"
            )
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """使用 half-split（半分离）RoPE 约定旋转 x 的最后一维。

    x   : (B, H, T, D)
    cos : (T, D)
    sin : (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 KV 头重复到和 query 头数一致。

    x: (B, n_kv_head, T, head_dim) -> (B, n_kv_head*n_rep, T, head_dim)
    """
    if n_rep == 1:
        return x
    b, n_kv_head, t, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv_head, n_rep, t, head_dim)
        .reshape(b, n_kv_head * n_rep, t, head_dim)
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        # 分开写投影，方便和 MiniMind/Llama 代码一一对应。
        self.q_proj = nn.Linear(
            config.n_embd, self.n_head * self.head_dim, bias=config.bias
        )
        self.k_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias
        )
        self.v_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias
        )
        self.o_proj = nn.Linear(
            self.n_head * self.head_dim, config.n_embd, bias=config.bias
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary = (
            RotaryEmbedding(self.head_dim, config.block_size, config.rope_theta)
            if config.use_rope
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ):
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            cos, sin = self.rotary(start_pos + t, x.device, q.dtype)
            cos = cos[start_pos : start_pos + t]
            sin = sin[start_pos : start_pos + t]
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # KV cache 保持 KV 头数；只有进入注意力计算时才复制。
        k_raw, v_raw = k, v
        present = (k_raw, v_raw) if use_cache else None
        k = repeat_kv(k_raw, self.n_rep)
        v = repeat_kv(v_raw, self.n_rep)

        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 因果 mask；有 KV cache 时也能正确处理。
        q_pos = torch.arange(start_pos, start_pos + t, device=x.device)
        k_pos = torch.arange(0, k.size(2), device=x.device)
        allowed = k_pos[None, :] <= q_pos[:, None]  # (t, t_k)
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_embd)
        out = self.resid_dropout(self.o_proj(out))
        return out, present


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.use_swiglu = config.use_swiglu
        if config.use_swiglu:
            # Llama 风格：隐藏维约为 8/3*d，向上取整到 256 的倍数。
            hidden = int(8 * config.n_embd / 3)
            hidden = ((hidden + 255) // 256) * 256
            self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        else:
            hidden = 4 * config.n_embd
            self.fc1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.fc2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)
            return self.dropout(self.down_proj(x))
        x = F.gelu(self.fc1(x))
        return self.dropout(self.fc2(self.dropout(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        norm_cls = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm1 = norm_cls(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = norm_cls(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ):
        attn_out, present = self.attn(
            self.norm1(x), past_kv=past_kv, use_cache=use_cache, start_pos=start_pos
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, present


class ModernGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        if not config.use_rope:
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.position_embedding = None

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        norm_cls = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm_f = norm_cls(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # GPT-2 风格：残差输出投影使用缩放初始化。
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ):
        b, t = idx.shape
        if t + start_pos > self.config.block_size:
            raise ValueError(
                f"序列长度 {t + start_pos} 超过 block_size {self.config.block_size}"
            )

        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            pos = torch.arange(start_pos, start_pos + t, device=idx.device)
            x = x + self.position_embedding(pos)
        x = self.drop(x)

        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_kv[i] if past_kv is not None else None
            x, present = block(
                x, past_kv=past, use_cache=use_cache, start_pos=start_pos
            )
            if use_cache:
                presents.append(present)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss, presents

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, use_cache: bool | None = None
    ):
        """采样 token；use_cache=True 时走 KV cache 解码路径。"""
        if use_cache is None:
            use_cache = self.config.use_cache

        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

        # 预填充 prompt，然后逐 token 使用 cache 解码。
        # 至少给第一个生成 token 留一个位置。
        max_prompt = self.config.block_size - 1
        idx_cond = idx[:, -max_prompt:]
        max_new_tokens = min(max_new_tokens, self.config.block_size - idx_cond.size(1))
        logits, _, past = self(idx_cond, use_cache=True, start_pos=0)
        last_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            idx_next = torch.multinomial(F.softmax(last_logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            logits, _, past = self(
                idx_next,
                past_kv=past,
                use_cache=True,
                start_pos=idx.shape[1] - 1,
            )
            last_logits = logits[:, -1, :]

        return idx
