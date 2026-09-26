"""L3: simple_gpt MLP SwiGLU silu(x)*y → gko.silu_mul (attention already F.sdpa)."""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _mlp_forward(self, x):
    a = self.c_fc1(x)
    b = self.c_fc2(x)
    try:
        fused = torch.ops.gko.silu_mul(a, b)
    except Exception:
        fused = F.silu(a) * b
    return self.c_proj(fused)


def apply(model) -> str:
    n = 0
    for m in model.modules():
        if type(m).__name__ != "MLP":
            continue
        if not (hasattr(m, "c_fc1") and hasattr(m, "c_fc2") and hasattr(m, "c_proj")):
            continue
        if getattr(m, "_gko_silu_mul", False):
            continue
        m._gko_orig_forward = m.forward
        m.forward = types.MethodType(_mlp_forward, m)
        m._gko_silu_mul = True
        n += 1
    return f"simple_gpt_silu_mul x{n}"
