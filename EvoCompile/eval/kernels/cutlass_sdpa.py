"""CUTLASS / Flash CUDA FMHA overlay (not Triton).

A100 libtorch already ships:
  - aten::_scaled_dot_product_efficient_attention  (CUTLASS mem-efficient FMHA)
  - aten::_scaled_dot_product_flash_attention       (FlashAttention CUDA)

Official compile(train_step) leaves extern cuBLAS bmm + Triton softmax.
This module replaces that leftover with the vendor fused CUDA kernel, using
the same per-case hooks as the Triton overlay.

Flash does not accept a non-null attn_mask, so 4D relative / pad / window
bias goes to CUTLASS mem-efficient. Causal with no mask can use either;
we prefer CUTLASS, Flash as fallback.

Gemma logit-softcap is not an FMHA epilogue — QK/PV still go through
cuBLAS/CUTLASS GEMM, tanh+softmax stay in PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def configure_cutlass_sdp():
    """Global flags so compiled graphs pick CUTLASS/Flash without a context manager."""
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    try:
        torch.backends.cuda.enable_cudnn_sdp(True)
    except Exception:
        pass


def _window_bias(n_q, n_k, window, device, dtype, causal):
    q_idx = torch.arange(n_q, device=device)[:, None]
    k_idx = torch.arange(n_k, device=device)[None, :]
    dist = k_idx - q_idx
    ok = (dist <= window) & (dist >= -window)
    if causal:
        ok = ok & (k_idx <= q_idx)
    neg = torch.finfo(dtype).min
    zeros = torch.zeros((), device=device, dtype=dtype)
    fill = torch.full((), neg, device=device, dtype=dtype)
    return torch.where(ok, zeros, fill)


def _as_mask(q, k, attn_bias, causal, window):
    """Return (attn_mask or None, is_causal). Flash needs mask is None for causal."""
    z, h, n_q, _ = q.shape
    n_k = int(k.size(-2))
    mask = None
    if attn_bias is not None and attn_bias.numel() > 0:
        b = attn_bias
        if b.dim() <= 2:
            mask = b.reshape(z, 1, 1, -1)[..., :n_k]
        elif b.dim() == 3:
            mask = b.unsqueeze(1)
        else:
            mask = b
            if mask.size(-1) != n_k:
                mask = mask[..., :n_k]
            if mask.size(-2) != n_q and mask.size(-2) == 1:
                mask = mask.expand(mask.size(0), mask.size(1), n_q, n_k)
    if window:
        w = _window_bias(n_q, n_k, int(window), q.device, q.dtype, bool(causal))
        mask = w if mask is None else mask + w
        causal = 0
    if mask is not None and bool(causal):
        # Fold causal into the additive mask so we can stay on CUTLASS (Flash
        # rejects non-null attn_mask).
        q_idx = torch.arange(n_q, device=q.device)[:, None]
        k_idx = torch.arange(n_k, device=q.device)[None, :]
        neg = torch.finfo(q.dtype).min
        causal_m = torch.where(
            k_idx <= q_idx,
            torch.zeros((), device=q.device, dtype=q.dtype),
            torch.full((), neg, device=q.device, dtype=q.dtype),
        )
        mask = mask + causal_m
        causal = 0
    if mask is not None:
        # CUTLASS mem-eff FMHA requires attn_mask.stride(-1) == 1. T5/mT5
        # relative bias is often a broadcast view (e.g. stride(-1)=n_heads).
        # contiguous() is a no-op when already packed.
        mask = mask.to(dtype=q.dtype).contiguous()
    return mask, bool(causal)


def _softcap_sdpa(q, k, v, mask, scale, softcap):
    scores = torch.matmul(q, k.transpose(-2, -1)) * float(scale)
    x = (scores / float(softcap)).clamp(-15.0, 15.0)
    scores = float(softcap) * torch.tanh(x)
    if mask is not None:
        scores = scores + mask
    p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


def cutlass_attention(q, k, v, attn_bias, scale, causal=0, window=0, softcap=0.0):
    squeeze = False
    if q.dim() == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        squeeze = True
    q = q.contiguous() if not q.is_contiguous() else q
    k = k.contiguous() if not k.is_contiguous() else k
    v = v.contiguous() if not v.is_contiguous() else v
    mask, is_causal = _as_mask(q, k, attn_bias, causal, window)
    if float(softcap or 0.0) > 0:
        out = _softcap_sdpa(q, k, v, mask, scale, softcap)
        return out.squeeze(1) if squeeze else out
    try:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=float(scale),
        )
    except RuntimeError:
        torch.backends.cuda.enable_math_sdp(True)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=float(scale),
        )
        torch.backends.cuda.enable_math_sdp(False)
    return out.squeeze(1) if squeeze else out
