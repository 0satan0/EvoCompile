"""Flash-style tiled attention: online softmax, no SxS HBM writeback.

Training used to fall back to F.sdpa inside autograd (extra vendor fwd+bwd on
top of the Triton fwd). That is why e2e train steps did not beat official
Dynamo: backward still materialized scores, and paid an extra attention fwd.

This file keeps a tiled Triton fwd and a matching tiled bwd (recompute P from
saved LSE, never write SxS). Ideas from Dao FlashAttention / the Triton tutorial.

Dynamo: use custom_op returning (out, lse) so compile(train_step) does not
graph-break / nan Autograd.Function.apply.
"""

from typing import Tuple

import torch

aten = torch.ops.aten

_KERNELS = None
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _flash_fwd(
        Q,
        K,
        V,
        Bias,
        Out,
        LSE,
        sm_scale,
        softcap,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        Z,
        H,
        N_Q,
        N_K,
        stride_bz,
        stride_bh,
        stride_bm,
        stride_bn,
        HAS_BIAS: tl.constexpr,
        HAS_BIAS_2D: tl.constexpr,
        CAUSAL: tl.constexpr,
        HAS_SOFTCAP: tl.constexpr,
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = (
            Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        )
        q_mask = offs_m[:, None] < N_Q
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1.0e6
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for start_n in range(0, N_K, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k_ptrs = (
                K
                + k_offset
                + (start_n + offs_n)[None, :] * stride_kn
                + offs_d[:, None] * stride_kk
            )
            v_ptrs = (
                V
                + v_offset
                + (start_n + offs_n)[:, None] * stride_vn
                + offs_d[None, :] * stride_vk
            )
            k_mask = (start_n + offs_n) < N_K
            k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0).to(q.dtype)
            qk = tl.dot(q, k)
            qk = qk.to(tl.float32) * sm_scale
            if HAS_BIAS_2D:
                b2 = tl.load(
                    Bias
                    + off_z * stride_bz
                    + off_h * stride_bh
                    + offs_m[:, None] * stride_bm
                    + (start_n + offs_n)[None, :] * stride_bn,
                    mask=q_mask & k_mask[None, :],
                    other=0.0,
                )
                qk = qk + b2
            elif HAS_BIAS:
                b_ptrs = Bias + off_z * N_K + (start_n + offs_n)
                bias = tl.load(b_ptrs, mask=k_mask, other=0.0)
                qk = qk + bias[None, :]
            if HAS_SOFTCAP:
                x = qk / softcap
                x = tl.minimum(tl.maximum(x, -15.0), 15.0)
                t = (1.0 - tl.exp(-2.0 * x)) / (1.0 + tl.exp(-2.0 * x))
                qk = softcap * t
            qk = tl.where(k_mask[None, :], qk, -1.0e6)
            if CAUSAL:
                qk = tl.where(
                    (start_n + offs_n)[None, :] <= offs_m[:, None],
                    qk,
                    -1.0e6,
                )
            if WINDOW > 0:
                dist = (start_n + offs_n)[None, :] - offs_m[:, None]
                qk = tl.where((dist <= WINDOW) & (dist >= -WINDOW), qk, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha[:, None]
            v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(q.dtype)
            acc += tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        l_i = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_i[:, None]
        o_ptrs = (
            Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        )
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)
        lse_ptrs = LSE + off_hz * N_Q + offs_m
        tl.store(lse_ptrs, m_i + tl.log(l_i), mask=offs_m < N_Q)

    @triton.jit
    def _flash_bwd_pre(
        O,
        DO,
        Delta,
        stride_oz,
        stride_oh,
        stride_om,
        stride_od,
        H,
        N_Q,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        o_off = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
        ptrs = O + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        mask = offs_m[:, None] < N_Q
        o = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        do = tl.load(
            DO + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        delta = tl.sum(o * do, 1)
        tl.store(Delta + off_hz * N_Q + offs_m, delta, mask=offs_m < N_Q)

    @triton.jit
    def _flash_bwd_dq(
        Q,
        K,
        V,
        Bias,
        DO,
        DQ,
        LSE,
        Delta,
        sm_scale,
        softcap,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dod,
        H,
        N_Q,
        N_K,
        stride_bz,
        stride_bh,
        stride_bm,
        stride_bn,
        HAS_BIAS: tl.constexpr,
        HAS_BIAS_2D: tl.constexpr,
        CAUSAL: tl.constexpr,
        HAS_SOFTCAP: tl.constexpr,
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        q_off = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_off = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_off = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        do_off = off_z.to(tl.int64) * stride_doz + off_h.to(tl.int64) * stride_doh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        q_mask = offs_m[:, None] < N_Q

        q = tl.load(
            Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
            mask=q_mask,
            other=0.0,
        )
        do = tl.load(
            DO + do_off + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
            mask=q_mask,
            other=0.0,
        )
        lse = tl.load(LSE + off_hz * N_Q + offs_m, mask=offs_m < N_Q, other=0.0)
        # Delta is inlined from O*dO (same M-tile). Avoids a third bwd launch.
        o = tl.load(
            Delta + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
            mask=q_mask,
            other=0.0,
        )
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
        dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for start_n in range(0, N_K, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k_mask = (start_n + offs_n) < N_K
            k = tl.load(
                K
                + k_off
                + (start_n + offs_n)[None, :] * stride_kn
                + offs_d[:, None] * stride_kk,
                mask=k_mask[None, :],
                other=0.0,
            ).to(q.dtype)
            v = tl.load(
                V
                + v_off
                + (start_n + offs_n)[None, :] * stride_vn
                + offs_d[:, None] * stride_vk,
                mask=k_mask[None, :],
                other=0.0,
            ).to(q.dtype)
            qk = tl.dot(q, k)
            qk = qk.to(tl.float32) * sm_scale
            if HAS_BIAS_2D:
                b2 = tl.load(
                    Bias
                    + off_z * stride_bz
                    + off_h * stride_bh
                    + offs_m[:, None] * stride_bm
                    + (start_n + offs_n)[None, :] * stride_bn,
                    mask=q_mask & k_mask[None, :],
                    other=0.0,
                )
                qk = qk + b2
            elif HAS_BIAS:
                bias = tl.load(
                    Bias + off_z * N_K + (start_n + offs_n),
                    mask=k_mask,
                    other=0.0,
                )
                qk = qk + bias[None, :]
            if HAS_SOFTCAP:
                x = qk / softcap
                x = tl.minimum(tl.maximum(x, -15.0), 15.0)
                t = (1.0 - tl.exp(-2.0 * x)) / (1.0 + tl.exp(-2.0 * x))
                qk = softcap * t
            qk = tl.where(k_mask[None, :], qk, -1.0e6)
            if CAUSAL:
                qk = tl.where(
                    (start_n + offs_n)[None, :] <= offs_m[:, None],
                    qk,
                    -1.0e6,
                )
            if WINDOW > 0:
                dist = (start_n + offs_n)[None, :] - offs_m[:, None]
                qk = tl.where((dist <= WINDOW) & (dist >= -WINDOW), qk, -1.0e6)
            p = tl.exp(qk - lse[:, None])
            dp = tl.dot(do.to(v.dtype), v)
            ds = p * (dp.to(tl.float32) - delta[:, None])
            k_nd = tl.load(
                K
                + k_off
                + (start_n + offs_n)[:, None] * stride_kn
                + offs_d[None, :] * stride_kk,
                mask=k_mask[:, None],
                other=0.0,
            )
            dq += tl.dot(ds.to(k_nd.dtype), k_nd) * sm_scale

        tl.store(
            DQ + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
            dq.to(DQ.dtype.element_ty),
            mask=q_mask,
        )

    @triton.jit
    def _flash_bwd_dkdv(
        Q,
        K,
        V,
        Bias,
        DO,
        DK,
        DV,
        LSE,
        Delta,
        sm_scale,
        softcap,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dod,
        H,
        N_Q,
        N_K,
        stride_bz,
        stride_bh,
        stride_bm,
        stride_bn,
        HAS_BIAS: tl.constexpr,
        HAS_BIAS_2D: tl.constexpr,
        CAUSAL: tl.constexpr,
        HAS_SOFTCAP: tl.constexpr,
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        start_n = tl.program_id(0) * BLOCK_N
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        q_off = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_off = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_off = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        do_off = off_z.to(tl.int64) * stride_doz + off_h.to(tl.int64) * stride_doh

        offs_n = start_n + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        k_mask = offs_n < N_K

        k = tl.load(
            K + k_off + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk,
            mask=k_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
            mask=k_mask[:, None],
            other=0.0,
        )
        dk = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        for start_m in range(0, N_Q, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            q_mask = (start_m + offs_m) < N_Q
            q = tl.load(
                Q
                + q_off
                + (start_m + offs_m)[:, None] * stride_qm
                + offs_d[None, :] * stride_qk,
                mask=q_mask[:, None],
                other=0.0,
            )
            do = tl.load(
                DO
                + do_off
                + (start_m + offs_m)[:, None] * stride_dom
                + offs_d[None, :] * stride_dod,
                mask=q_mask[:, None],
                other=0.0,
            )
            lse = tl.load(
                LSE + off_hz * N_Q + (start_m + offs_m),
                mask=q_mask,
                other=0.0,
            )
            o = tl.load(
                Delta
                + q_off
                + (start_m + offs_m)[:, None] * stride_qm
                + offs_d[None, :] * stride_qk,
                mask=q_mask[:, None],
                other=0.0,
            )
            delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
            qk = tl.dot(q, k.to(q.dtype))
            qk = qk.to(tl.float32) * sm_scale
            if HAS_BIAS_2D:
                b2 = tl.load(
                    Bias
                    + off_z * stride_bz
                    + off_h * stride_bh
                    + (start_m + offs_m)[:, None] * stride_bm
                    + offs_n[None, :] * stride_bn,
                    mask=q_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                qk = qk + b2
            elif HAS_BIAS:
                bias = tl.load(Bias + off_z * N_K + offs_n, mask=k_mask, other=0.0)
                qk = qk + bias[None, :]
            if HAS_SOFTCAP:
                x = qk / softcap
                x = tl.minimum(tl.maximum(x, -15.0), 15.0)
                t = (1.0 - tl.exp(-2.0 * x)) / (1.0 + tl.exp(-2.0 * x))
                qk = softcap * t
            qk = tl.where(k_mask[None, :], qk, -1.0e6)
            if CAUSAL:
                qk = tl.where(
                    offs_n[None, :] <= (start_m + offs_m)[:, None],
                    qk,
                    -1.0e6,
                )
            if WINDOW > 0:
                dist = offs_n[None, :] - (start_m + offs_m)[:, None]
                qk = tl.where((dist <= WINDOW) & (dist >= -WINDOW), qk, -1.0e6)
            qk = tl.where(q_mask[:, None], qk, -1.0e6)
            p = tl.exp(qk - lse[:, None])
            p = tl.where(q_mask[:, None], p, 0.0)
            dv += tl.dot(tl.trans(p.to(do.dtype), 1, 0), do)
            dp = tl.dot(do.to(v.dtype), tl.trans(v, 1, 0))
            ds = p * (dp.to(tl.float32) - delta[:, None])
            ds = tl.where(q_mask[:, None], ds, 0.0)
            dk += tl.dot(tl.trans(q, 1, 0), ds.to(q.dtype)) * sm_scale

        tl.store(
            DK + k_off + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk,
            dk.to(DK.dtype.element_ty),
            mask=k_mask[None, :],
        )
        tl.store(
            DV + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
            dv.to(DV.dtype.element_ty),
            mask=k_mask[:, None],
        )

    _KERNELS = {
        "fwd": _flash_fwd,
        "pre": _flash_bwd_pre,
        "dq": _flash_bwd_dq,
        "dkdv": _flash_bwd_dkdv,
    }
except Exception:
    _KERNELS = None


def _fallback(q, k, v, attn_bias, scale, causal=0):
    has = attn_bias is not None and attn_bias.numel() > 0
    if causal and not has:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, scale=scale, is_causal=True
        )
    mask = None
    if has:
        mask = attn_bias.to(dtype=q.dtype)
        while mask.dim() < 4:
            mask = mask.unsqueeze(1)
        if causal:
            s = q.size(-2)
            q_idx = torch.arange(s, device=q.device)[:, None]
            k_idx = torch.arange(s, device=q.device)[None, :]
            neg = torch.finfo(q.dtype).min
            causal_m = torch.where(
                k_idx <= q_idx,
                torch.zeros((), device=q.device, dtype=q.dtype),
                torch.full((), neg, device=q.device, dtype=q.dtype),
            )
            mask = mask + causal_m
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale
    )


def _can_triton(q, k=None):
    if _KERNELS is None or not q.is_cuda or q.dim() != 4:
        return False
    if k is not None and (k.dim() != 4 or k.size(-1) != q.size(-1)):
        return False
    d = int(q.size(-1))
    n_q = int(q.size(-2))
    n_k = int(k.size(-2)) if k is not None else n_q
    return d in (32, 64, 128, 256) and n_q >= 1 and n_k >= 1


def _bias_pack(q, k, attn_bias):
    """Return (bias, has_1d, has_2d).

    1D is key-padding [Z, N_K]. 2D is relative / full mask [Z, H, N_Q, N_K].
    """
    z, h, n_q, _ = q.shape
    n_k = int(k.size(-2))
    dummy = q.new_zeros((z, n_k))
    if attn_bias is None or attn_bias.numel() == 0:
        return dummy, False, False
    b = attn_bias
    if b.dim() <= 2 and b.numel() == z * n_k:
        return b.reshape(z, n_k).contiguous(), True, False
    if b.dim() == 3 and b.size(-1) == n_k and b.numel() == z * n_k:
        return b.reshape(z, n_k).contiguous(), True, False
    if b.dim() == 4 and b.size(-2) == 1 and b.size(-1) == n_k:
        row = b[:, 0, 0, :] if b.size(0) == z else b[0, 0, 0, :].expand(z, n_k)
        return row.reshape(z, n_k).contiguous(), True, False
    if b.dim() == 4 and b.size(-2) == n_q and b.size(-1) == n_k:
        # Keep a broadcast view. contiguous() here would write the full SxS
        # tensor — the opposite of tiled FA.
        if b.size(0) == 1:
            b = b.expand(z, b.size(1), n_q, n_k)
        elif b.size(0) != z and z % int(b.size(0)) == 0:
            # Swin shifted-window mask is [nW, H, N, N]; q is [B*nW, H, N, D].
            b = b.repeat(z // int(b.size(0)), 1, 1, 1)
        if b.size(1) == 1:
            b = b.expand(b.size(0), h, n_q, n_k)
        return b, False, True
    return dummy, False, False


def _maybe_contig(t):
    return t if t.is_contiguous() else t.contiguous()


def _bias_strides(bias, has_2d):
    if has_2d:
        return (
            int(bias.stride(0)),
            int(bias.stride(1)),
            int(bias.stride(2)),
            int(bias.stride(3)),
        )
    return 0, 0, 0, 1


def _tile_sizes(n_k, d, bwd=False):
    # d=256 (Gemma) needs small tiles. BLOCK_N=128 in bwd tripped Llama's tl.trans.
    if d >= 256:
        return 32, 32
    if n_k <= 64:
        return 64, 64
    if (not bwd) and n_k <= 128 and d <= 64:
        return 64, 128
    return 64, 64


def _kernel_meta(causal, window, softcap, d, bias_2d=False):
    return dict(
        HAS_SOFTCAP=float(softcap) > 0,
        WINDOW=int(window) if window else 0,
        CAUSAL=bool(causal),
        num_warps=4 if d <= 64 else 8,
        num_stages=1 if (d >= 256 or bias_2d) else 2,
    )


def _triton_fwd(q, k, v, attn_bias, scale, causal=0, window=0, softcap=0.0):
    q = _maybe_contig(q)
    k = _maybe_contig(k)
    v = _maybe_contig(v)
    z, h, n_q, d = q.shape
    n_k = int(k.size(-2))
    out = torch.empty_like(q)
    lse = torch.empty((z * h, n_q), device=q.device, dtype=torch.float32)
    bias, has_bias, has_bias_2d = _bias_pack(q, k, attn_bias)
    BLOCK_M, BLOCK_N = _tile_sizes(n_k, d)
    if has_bias_2d and n_k > 128:
        BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(n_q, BLOCK_M), z * h)
    meta = _kernel_meta(causal, window, softcap, d, has_bias_2d)
    _KERNELS["fwd"][grid](
        q,
        k,
        v,
        bias,
        out,
        lse,
        float(scale),
        float(softcap),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        z,
        h,
        n_q,
        n_k,
        *_bias_strides(bias, has_bias_2d),
        HAS_BIAS=has_bias,
        HAS_BIAS_2D=has_bias_2d,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=d,
        **meta,
    )
    return out, lse


def _triton_bwd(
    q, k, v, attn_bias, out, lse, dout, scale, causal=0, window=0, softcap=0.0
):
    q = _maybe_contig(q)
    k = _maybe_contig(k)
    v = _maybe_contig(v)
    out = _maybe_contig(out)
    dout = _maybe_contig(dout)
    if tuple(out.stride()) != tuple(q.stride()):
        out = out.contiguous()
    z, h, n_q, d = q.shape
    n_k = int(k.size(-2))
    bias, has_bias, has_bias_2d = _bias_pack(q, k, attn_bias)
    BLOCK_M, BLOCK_N = _tile_sizes(n_k, d, bwd=True)
    if has_bias_2d and n_k > 128:
        BLOCK_M, BLOCK_N = 64, 64
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    grid_m = (triton.cdiv(n_q, BLOCK_M), z * h)
    grid_n = (triton.cdiv(n_k, BLOCK_N), z * h)
    meta = _kernel_meta(causal, window, softcap, d, has_bias_2d)
    kw = dict(num_warps=meta.pop("num_warps"), num_stages=meta.pop("num_stages"))
    _KERNELS["dq"][grid_m](
        q,
        k,
        v,
        bias,
        dout,
        dq,
        lse,
        out,
        float(scale),
        float(softcap),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        h,
        n_q,
        n_k,
        *_bias_strides(bias, has_bias_2d),
        HAS_BIAS=has_bias,
        HAS_BIAS_2D=has_bias_2d,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=d,
        **meta,
        **kw,
    )
    _KERNELS["dkdv"][grid_n](
        q,
        k,
        v,
        bias,
        dout,
        dk,
        dv,
        lse,
        out,
        float(scale),
        float(softcap),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        h,
        n_q,
        n_k,
        *_bias_strides(bias, has_bias_2d),
        HAS_BIAS=has_bias,
        HAS_BIAS_2D=has_bias_2d,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=d,
        **meta,
        **kw,
    )
    return dq, dk, dv

class _FusedSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_bias, scale, causal):
        out, lse = _triton_fwd(q, k, v, attn_bias, float(scale), int(causal))
        ctx.save_for_backward(q, k, v, attn_bias, out, lse)
        ctx.scale = float(scale)
        ctx.causal = int(causal)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, attn_bias, out, lse = ctx.saved_tensors
        dq, dk, dv = _triton_bwd(
            q, k, v, attn_bias, out, lse, dout, ctx.scale, ctx.causal
        )
        return dq, dk, dv, None, None, None


def fused_sdpa_forward(q, k, v, attn_bias, scale, causal=0):
    if not _can_triton(q, k):
        return _fallback(q, k, v, attn_bias, scale, causal)
    return _triton_fwd(q, k, v, attn_bias, scale, causal)[0]


def fused_attention(q, k, v, attn_bias, scale, causal=0, window=0, softcap=0.0):
    """Differentiable fused attention. Triton or CUTLASS FMHA; Dynamo-safe."""
    import os

    backend = os.environ.get("GKO_ATTN_BACKEND", "triton").strip().lower()
    if backend in ("auto", "hybrid"):
        from kernels.attn_policy import resolved

        backend = resolved()
    if backend in ("cutlass", "fmha", "sdpa"):
        from kernels.cutlass_gemm_attn import handwritten_attention

        return handwritten_attention(
            q, k, v, attn_bias, scale, causal, window, softcap
        )
    squeeze = False
    if q.dim() == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        squeeze = True
    if attn_bias is None:
        # Empty so _bias_pack keeps HAS_BIAS=False. A dense zero pad would
        # enable the 1D bias path and extra loads for causal Llama/Gemma.
        attn_bias = q.new_empty((0,))
    q = q.contiguous() if not q.is_contiguous() else q
    k = k.contiguous() if not k.is_contiguous() else k
    v = v.contiguous() if not v.is_contiguous() else v
    if attn_bias is not None and not attn_bias.is_contiguous() and attn_bias.dim() <= 2:
        attn_bias = attn_bias.contiguous()
    if not _can_triton(q, k):
        out = _fallback(q, k, v, attn_bias, scale, causal)
        return out.squeeze(1) if squeeze else out
    _ensure_op()
    out, _lse = torch.ops.gko.fused_sdpa_pair(
        q,
        k,
        v,
        attn_bias,
        float(scale),
        int(causal),
        int(window or 0),
        float(softcap or 0.0),
    )
    return out.squeeze(1) if squeeze else out


def _setup_pair(ctx, inputs, output):
    q, k, v, attn_bias, scale, causal, window, softcap = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, attn_bias, out, lse)
    ctx.scale = float(scale)
    ctx.causal = int(causal)
    ctx.window = int(window)
    ctx.softcap = float(softcap)


def _bwd_pair(ctx, dout, dlse):
    # Inductor compiles this Python backward. Launching Triton here is traced
    # as FakeTensor.data_ptr and blows up. Call an opaque bwd custom_op instead.
    q, k, v, attn_bias, out, lse = ctx.saved_tensors
    dq, dk, dv = torch.ops.gko.fused_sdpa_bwd(
        q,
        k,
        v,
        attn_bias,
        out,
        lse,
        dout,
        float(ctx.scale),
        int(ctx.causal),
        int(ctx.window),
        float(ctx.softcap),
    )
    return dq, dk, dv, None, None, None, None, None


def _setup_context(ctx, inputs, output):
    q, k, v, attn_bias, scale, causal = inputs
    ctx.save_for_backward(q, k, v, attn_bias, output)
    ctx.scale = float(scale)
    ctx.causal = int(causal)


def _backward(ctx, grad_out):
    """Catalog custom_op bwd: recompute LSE with Triton fwd, then Triton bwd.

    Still one extra fwd vs Autograd.Function, but no vendor SDPA recompute.
    """
    q, k, v, attn_bias, out = ctx.saved_tensors
    if not _can_triton(q, k):
        qn = q.detach().requires_grad_(True)
        kn = k.detach().requires_grad_(True)
        vn = v.detach().requires_grad_(True)
        with torch.enable_grad():
            y = _fallback(qn, kn, vn, attn_bias, ctx.scale, ctx.causal)
            y.backward(grad_out)
        return qn.grad, kn.grad, vn.grad, None, None, None
    _, lse = _triton_fwd(q, k, v, attn_bias, ctx.scale, ctx.causal)
    dq, dk, dv = _triton_bwd(
        q, k, v, attn_bias, out, lse, grad_out, ctx.scale, ctx.causal
    )
    return dq, dk, dv, None, None, None


_OP_READY = False
_AUTOGRADED = False


def _ensure_op():
    global _OP_READY, _AUTOGRADED
    if not _OP_READY:
        if not (
            hasattr(torch.ops, "gko") and hasattr(torch.ops.gko, "fused_sdpa_pair")
        ):

            @torch.library.custom_op("gko::fused_sdpa_pair", mutates_args=())
            def fused_sdpa_pair(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_bias: torch.Tensor,
                scale: float,
                causal: int,
                window: int,
                softcap: float,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                if not _can_triton(q, k):
                    y = _fallback(q, k, v, attn_bias, scale, causal)
                    lse = torch.zeros(
                        (q.size(0) * q.size(1), q.size(2)),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    return y, lse
                return _triton_fwd(
                    q, k, v, attn_bias, scale, causal, window, softcap
                )

            @fused_sdpa_pair.register_fake
            def _(q, k, v, attn_bias, scale, causal, window, softcap):
                # Real kernel contiguous()-copies QKV and writes a dense out.
                # empty_like(q) keeps transpose strides; Inductor then fails
                # assert_size_stride.
                lse = q.new_empty(
                    (q.size(0) * q.size(1), q.size(2)), dtype=torch.float32
                )
                return q.new_empty(q.shape), lse

        if not (hasattr(torch.ops, "gko") and hasattr(torch.ops.gko, "fused_sdpa_bwd")):

            @torch.library.custom_op("gko::fused_sdpa_bwd", mutates_args=())
            def fused_sdpa_bwd(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_bias: torch.Tensor,
                out: torch.Tensor,
                lse: torch.Tensor,
                dout: torch.Tensor,
                scale: float,
                causal: int,
                window: int,
                softcap: float,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return _triton_bwd(
                    q, k, v, attn_bias, out, lse, dout, scale, causal, window, softcap
                )

            @fused_sdpa_bwd.register_fake
            def _(q, k, v, attn_bias, out, lse, dout, scale, causal, window, softcap):
                return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)

        try:
            torch.library.register_autograd(
                "gko::fused_sdpa_pair",
                _bwd_pair,
                setup_context=_setup_pair,
            )
        except RuntimeError as e:
            if "already" not in str(e).lower():
                raise

        if not (hasattr(torch.ops, "gko") and hasattr(torch.ops.gko, "fused_sdpa")):

            @torch.library.custom_op("gko::fused_sdpa", mutates_args=())
            def fused_sdpa(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_bias: torch.Tensor,
                scale: float,
                causal: int,
            ) -> torch.Tensor:
                return fused_sdpa_forward(q, k, v, attn_bias, scale, causal)

            @fused_sdpa.register_fake
            def _(q, k, v, attn_bias, scale, causal):
                return q.new_empty(q.shape)

        _OP_READY = True
    if not _AUTOGRADED:
        try:
            torch.library.register_autograd(
                "gko::fused_sdpa",
                _backward,
                setup_context=_setup_context,
            )
        except RuntimeError as e:
            if "already" not in str(e).lower():
                raise
        _AUTOGRADED = True


def register(backend=None):
    import os

    if backend:
        os.environ["GKO_ATTN_BACKEND"] = str(backend)
    b = os.environ.get("GKO_ATTN_BACKEND", "").strip().lower()
    if b == "pipe":
        os.environ["GKO_CUTLASS_HANDWRITTEN"] = "1"
        os.environ["GKO_ATTN_BACKEND"] = "cutlass"
        b = "cutlass"
    if b in ("cutlass", "fmha", "sdpa", "auto", "hybrid"):
        from kernels.cutlass_sdpa import configure_cutlass_sdp

        configure_cutlass_sdp()
        _ensure_op()
        if b in ("cutlass", "fmha", "sdpa", "auto", "hybrid"):
            try:
                from kernels.cutlass_gemm_attn import _ensure_op as _ensure_cutlass

                _ensure_cutlass()
            except Exception:
                pass
        if b in ("cutlass", "fmha", "sdpa"):
            return
    _ensure_op()


def _sync_ms(fn, warmup, iters):
    import time

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def microbench(
    *,
    z=128,
    h=12,
    s=128,
    d=64,
    dtype=torch.float16,
    warmup=10,
    iters=50,
    train=False,
    causal=0,
):
    """DistilBert-like: AOT had bmm f16[1536,128,128] = (B*H,S,S) with B*H=1536."""
    register()
    device = "cuda"
    scale = d ** -0.5
    bias = torch.zeros(z, s, device=device, dtype=dtype)

    def make():
        q = torch.randn(z, h, s, d, device=device, dtype=dtype)
        k = torch.randn(z, h, s, d, device=device, dtype=dtype)
        v = torch.randn(z, h, s, d, device=device, dtype=dtype)
        if train:
            q, k, v = [t.requires_grad_(True) for t in (q, k, v)]
        return q, k, v

    q0, k0, v0 = make()

    def unfused(q, k, v):
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if causal:
            idx = torch.arange(s, device=q.device)
            scores = scores.masked_fill(idx[None, None, None, :] > idx[None, None, :, None], -65504.0)
        p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(p, v)

    def sdpa(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=bool(causal)
        )

    def triton_op(q, k, v):
        return fused_attention(q, k, v, bias, float(scale), int(causal))

    def wrap(fn, q, k, v):
        def run():
            if train:
                if q.grad is not None:
                    q.grad = None
                    k.grad = None
                    v.grad = None
                out = fn(q, k, v)
                out.square().mean().backward()
                return out
            return fn(q, k, v)

        return run

    out = {}
    for name, fn in (("unfused_bmm", unfused), ("in_tree_sdpa", sdpa), ("triton_flash", triton_op)):
        q, k, v = make()
        out[name] = _sync_ms(wrap(fn, q, k, v), warmup, iters)
    with torch.no_grad():
        ref = unfused(q0, k0, v0)
        try:
            out["triton_max_abs"] = float((triton_op(q0, k0, v0) - ref).abs().max())
        except Exception as e:
            out["triton_err"] = str(e)
        try:
            out["sdpa_max_abs"] = float((sdpa(q0, k0, v0) - ref).abs().max())
        except Exception as e:
            out["sdpa_err"] = str(e)
    if train:
        q, k, v = [t.detach().requires_grad_(True) for t in (q0, k0, v0)]
        unfused(q, k, v).square().mean().backward()
        rq, rk, rv = q.grad.detach(), k.grad.detach(), v.grad.detach()
        q, k, v = [t.detach().requires_grad_(True) for t in (q0, k0, v0)]
        triton_op(q, k, v).square().mean().backward()
        out["triton_dQ_abs"] = float((q.grad - rq).abs().max())
        out["triton_dK_abs"] = float((k.grad - rk).abs().max())
        out["triton_dV_abs"] = float((v.grad - rv).abs().max())
    out["train"] = int(train)
    out["causal"] = int(causal)
    return out
