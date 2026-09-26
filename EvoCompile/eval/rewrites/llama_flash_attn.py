"""L3: llama Attention manual matmul+softmax → F.scaled_dot_product_attention only.

Does not touch FeedForward (use replace_pattern silu_mul FX for SwiGLU).
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F


def _llama_attn_forward(self, x, start_pos, freqs_cis, mask):
    from torchbenchmark.models.llama.model import apply_rotary_emb

    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    self.cache_k = self.cache_k.to(xq)
    self.cache_v = self.cache_v.to(xq)
    with torch.no_grad():
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

    keys = self.cache_k[:bsz, : start_pos + seqlen]
    values = self.cache_v[:bsz, : start_pos + seqlen]

    q = xq.transpose(1, 2)
    k = keys.transpose(1, 2)
    v = values.transpose(1, 2)
    scale = 1.0 / math.sqrt(self.head_dim)
    # Match upstream (mask commented out); is_causal breaks Dynamo with cache_len!=q_len.
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, scale=scale, is_causal=False
    )
    out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(out)


def apply(model) -> str:
    n_attn = 0
    for m in model.modules():
        name = type(m).__name__
        mod = type(m).__module__ or ""
        if name == "Attention" and "llama" in mod and hasattr(m, "wq") and hasattr(m, "cache_k"):
            if getattr(m, "_gko_llama_flash", False):
                continue
            m._gko_orig_forward = m.forward
            m.forward = types.MethodType(_llama_attn_forward, m)
            m._gko_llama_flash = True
            n_attn += 1
    return f"llama_flash_attn x{n_attn} attn"
