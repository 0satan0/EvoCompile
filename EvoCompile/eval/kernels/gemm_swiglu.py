"""GEMM + SwiGLU epilogue: one kernel for gate/up projection + silu*mul.

Unlike standalone ``silu_mul`` / Titan ``silu_and_mul`` (opaque elementwise after
a separate GEMM), this fuses the activation into the matmul epilogue so the
2H intermediate is never fully written then re-read for SiLU.

Layout: ``weight`` is ``(2H, D)`` (gate then up, matching ``F.linear``),
``x`` is ``(..., D)``, output is ``(..., H)`` = silu(x@W_gate) * (x@W_up).

On A100, Triton ``tl.dot`` often loses to cuBLAS for large K; set
``GKO_GEMM_SWIGLU_BACKEND=cublas`` to fall back to cuBLAS linear + stock silu*mul
(Inductor-friendly). ``auto`` (default) tries Triton then falls back.
CUTLASS tensor-op + SwiGLU epilogue is the production path if available
(``act`` would need to emit half-width output); not wired until the .cu grows
a SwiGLU epilogue (current cutlass helper only has gelu/relu).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_KERNEL = None
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _gemm_swiglu_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """A[M,K] @ B[K,2N] -> C[M,N] with SwiGLU over the 2N columns.

        B columns are packed [gate | up]; each program covers BLOCK_N of the
        H (=N) output, loading both gate and up tiles in the N dimension.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        # gate columns: [0, N), up columns: [N, 2N)
        b_gate_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_up_ptrs = b_ptr + offs_k[:, None] * stride_bk + (offs_n + N)[None, :] * stride_bn

        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_off = k * BLOCK_K
            k_mask = (offs_k + k_off) < K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & k_mask[None, :],
                other=0.0,
            )
            bg = tl.load(
                b_gate_ptrs,
                mask=k_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )
            bu = tl.load(
                b_up_ptrs,
                mask=k_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )
            acc_g += tl.dot(a, bg)
            acc_u += tl.dot(a, bu)
            a_ptrs += BLOCK_K * stride_ak
            b_gate_ptrs += BLOCK_K * stride_bk
            b_up_ptrs += BLOCK_K * stride_bk

        # silu(gate) * up
        sig = 1.0 / (1.0 + tl.exp(-acc_g))
        out = (acc_g * sig) * acc_u
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(
            c_ptrs,
            out.to(c_ptr.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    _KERNEL = _gemm_swiglu_kernel
except Exception:
    _KERNEL = None


def _eager_gemm_swiglu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """weight (2H, D); out (..., H)."""
    gate_up = F.linear(x, weight)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _triton_gemm_swiglu(x2d: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
    """x2d (M, D), weight_t (D, 2H) column-major for B in A@B; out (M, H)."""
    if _KERNEL is None or not x2d.is_cuda:
        w = weight_t.t().contiguous()
        return _eager_gemm_swiglu(x2d, w)
    M, K = x2d.shape
    K2, two_h = weight_t.shape
    if K != K2 or two_h % 2:
        raise ValueError(f"gemm_swiglu shape mismatch x={x2d.shape} wT={weight_t.shape}")
    H = two_h // 2
    x2d = x2d.contiguous()
    weight_t = weight_t.contiguous()
    out = torch.empty((M, H), device=x2d.device, dtype=x2d.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
    _KERNEL[grid](
        x2d,
        weight_t,
        out,
        M,
        H,
        K,
        x2d.stride(0),
        x2d.stride(1),
        weight_t.stride(0),
        weight_t.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


def gemm_swiglu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fused gate/up linear + SwiGLU.

    Args:
        x: ``(..., D)``
        weight: ``(2H, D)`` — gate rows then up rows (``F.linear`` layout)
    Returns:
        ``(..., H)``
    """
    backend = os.environ.get("GKO_GEMM_SWIGLU_BACKEND", "auto").strip().lower()
    if backend == "cublas" or _KERNEL is None or not x.is_cuda:
        return _eager_gemm_swiglu(x, weight)
    lead = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    # A@B with B = weight.T → (D, 2H)
    try:
        out2d = _triton_gemm_swiglu(x2d, weight.t())
    except Exception:
        return _eager_gemm_swiglu(x, weight)
    if backend == "triton":
        return out2d.reshape(lead + (out2d.shape[-1],))
    # auto: keep Triton result (caller benches vs cublas)
    return out2d.reshape(lead + (out2d.shape[-1],))


def _setup_context(ctx, inputs, output):
    x, weight = inputs
    ctx.save_for_backward(x, weight)


def _backward(ctx, grad_out):
    x, weight = ctx.saved_tensors
    # Recompute gate_up for correct grads (same as stock SwiGLU via linear).
    gate_up = F.linear(x, weight)
    gate, up = gate_up.chunk(2, dim=-1)
    sig = torch.sigmoid(gate)
    silu_g = gate * sig
    # dL/dup = dL/dy * silu(gate); dL/dgate = dL/dy * up * dsilu
    d_up = grad_out * silu_g
    d_silu = grad_out * up
    d_gate = d_silu * (sig * (1 + gate * (1 - sig)))
    d_gate_up = torch.cat([d_gate, d_up], dim=-1)
    needs = getattr(ctx, "needs_input_grad", None) or (True, True)
    gx = None
    gw = None
    if needs[0]:
        gx = F.linear(d_gate_up, weight.t())
        gx = gx.reshape(x.shape)
    if needs[1]:
        # weight (2H, D): grad = d_gate_up.T @ x_flat
        xf = x.reshape(-1, x.shape[-1])
        dg = d_gate_up.reshape(-1, d_gate_up.shape[-1])
        gw = dg.t().mm(xf)
    return gx, gw


_OP_READY = False
_AUTOGRADED = False
_PASS_INSTALLED = False


def _ensure_op():
    global _OP_READY, _AUTOGRADED
    if not _OP_READY:
        if not (hasattr(torch.ops, "gko") and hasattr(torch.ops.gko, "gemm_swiglu")):

            @torch.library.custom_op("gko::gemm_swiglu", mutates_args=())
            def _op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
                return gemm_swiglu(x, weight)

            @_op.register_fake
            def _(x, weight):
                h = weight.shape[0] // 2
                return x.new_empty(*x.shape[:-1], h)

        _OP_READY = True
    if not _AUTOGRADED:
        try:
            torch.library.register_autograd(
                "gko::gemm_swiglu",
                _backward,
                setup_context=_setup_context,
            )
        except Exception:
            pass
        _AUTOGRADED = True


def register(backend: str = "") -> None:
    """Install custom op. No FX pass — Titan hooks call this from FusedSwiGLU."""
    global _PASS_INSTALLED
    if backend:
        os.environ["GKO_GEMM_SWIGLU_BACKEND"] = backend
    _ensure_op()
    _PASS_INSTALLED = True


def apply(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Public entry used by Titan override; prefers registered custom op."""
    _ensure_op()
    try:
        return torch.ops.gko.gemm_swiglu(x, weight)
    except Exception:
        return gemm_swiglu(x, weight)
