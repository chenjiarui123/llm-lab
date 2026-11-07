"""ModernGPT 实验模型的正确性检查（中文注释版）。

运行：
    python tests.py
"""

from __future__ import annotations

import torch

from model import (
    GPTConfig,
    ModernGPT,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_emb,
)


def test_rmsnorm():
    x = torch.randn(2, 5, 16)
    y = RMSNorm(16)(x)
    assert y.shape == x.shape
    # weight=1 时，最后一维的均方值应该约为 1。
    ms = y.pow(2).mean(dim=-1)
    assert torch.allclose(ms, torch.ones_like(ms), atol=1e-4), ms
    print("ok: RMSNorm")


def test_rope():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    q = torch.randn(1, 2, 4, 8)
    cos, sin = rope(4, torch.device("cpu"), q.dtype)
    q_rot = apply_rotary_emb(q, cos, sin)
    assert q_rot.shape == q.shape
    # 旋转保持每个二维对/整个向量的范数不变。
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    print("ok: RoPE")


def build_model(**kwargs):
    config = GPTConfig(
        vocab_size=17,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
        **kwargs,
    )
    return ModernGPT(config)


def test_forward_switches():
    variants = [
        {},
        {"use_rmsnorm": True},
        {"use_swiglu": True},
        {"use_rope": True},
        {"use_rmsnorm": True, "use_rope": True},
        {"use_rmsnorm": True, "use_swiglu": True, "use_rope": True},
        {"use_rmsnorm": True, "use_swiglu": True, "use_rope": True, "n_kv_head": 2},
    ]
    x = torch.randint(0, 17, (2, 8))
    for kwargs in variants:
        model = build_model(**kwargs)
        logits, loss, _ = model(x, x)
        assert logits.shape == (2, 8, 17), (kwargs, logits.shape)
        assert loss is not None and loss.item() > 0
        print(f"ok: forward {kwargs}")


def test_gqa_shapes():
    model = build_model(n_kv_head=2)
    attn = model.blocks[0].attn
    assert attn.n_head == 4
    assert attn.n_kv_head == 2
    assert attn.n_rep == 2
    assert attn.k_proj.out_features == 2 * attn.head_dim
    assert attn.v_proj.out_features == 2 * attn.head_dim
    print("ok: GQA shapes")


def test_position_switches():
    baseline = build_model()
    assert baseline.position_embedding is not None
    rope = build_model(use_rope=True)
    assert rope.position_embedding is None
    print("ok: position embedding switches")


def test_cache_equivalence():
    torch.manual_seed(0)
    model = build_model(
        use_rmsnorm=True,
        use_swiglu=True,
        use_rope=True,
        n_kv_head=2,
    ).eval()

    idx = torch.randint(0, 17, (2, 6))
    logits_full, _, _ = model(idx)
    logits_prefill, _, past = model(idx, use_cache=True, start_pos=0)
    assert torch.allclose(logits_full, logits_prefill, atol=1e-5)

    next_idx = torch.randint(0, 17, (2, 1))
    logits_cached, _, _ = model(
        next_idx, past_kv=past, use_cache=True, start_pos=6
    )
    logits_full_next, _, _ = model(torch.cat((idx, next_idx), dim=1))
    assert torch.allclose(
        logits_cached[:, -1, :], logits_full_next[:, -1, :], atol=1e-5
    )
    print("ok: KV cache equivalence")


def test_generate_cache():
    model = build_model(
        use_rmsnorm=True,
        use_swiglu=True,
        use_rope=True,
        n_kv_head=2,
        use_cache=True,
    ).eval()
    idx = torch.randint(0, 17, (1, 4))
    out_cache = model.generate(idx, max_new_tokens=3, use_cache=True)
    out_plain = model.generate(idx, max_new_tokens=3, use_cache=False)
    assert out_cache.shape == (1, 7)
    assert out_plain.shape == (1, 7)
    print("ok: generate with and without cache")


if __name__ == "__main__":
    test_rmsnorm()
    test_rope()
    test_forward_switches()
    test_gqa_shapes()
    test_position_switches()
    test_cache_equivalence()
    test_generate_cache()
    print("\nall tests passed")
