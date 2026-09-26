"""Load the handwritten CUTLASS GEMM+epilogue / tiled FMHA extension."""
import os
from pathlib import Path
from typing import Tuple

import torch

_EXT = None
_FAIL = None

_SRC = Path(__file__).resolve().parent / "cutlass_fmha" / "gko_cutlass_attn.cu"


def cutlass_home():
    for p in (
        os.environ.get("GKO_CUTLASS_HOME", ""),
        os.environ.get("CUTLASS_HOME", ""),
        str(Path(__file__).resolve().parents[2] / "third_party" / "cutlass"),
        "/usr/local/cutlass",
    ):
        if p and (Path(p) / "include" / "cutlass" / "cutlass.h").is_file():
            return p
    return ""


def load_ext():
    global _EXT, _FAIL
    if _EXT is not None:
        return _EXT
    if _FAIL is not None:
        raise _FAIL
    home = cutlass_home()
    if not home:
        _FAIL = RuntimeError("CUTLASS headers not found (GKO_CUTLASS_HOME)")
        raise _FAIL
    from torch.utils.cpp_extension import load

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    os.environ.setdefault("CUDA_HOME", cuda_home)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    # Host 68 /usr/local/bin/ninja is a pip stub that hangs on --version.
    os.environ["PATH"] = cuda_home + "/bin:/usr/bin:/bin"
    extra = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "-gencode=arch=compute_80,code=sm_80",
        "-I" + str(Path(home) / "include"),
    ]
    try:
        _EXT = load(
            name="gko_cutlass_attn_pipe2",
            sources=[str(_SRC)],
            extra_cuda_cflags=extra,
            extra_cflags=["-O3", "-std=c++17"],
            extra_include_paths=[str(Path(home) / "include")],
            verbose=True,
            with_cuda=True,
        )
    except Exception as e:
        _FAIL = e
        raise
    return _EXT


def can_handwritten(q):
    if q is None or q.dim() != 4:
        return False
    if q.dtype != torch.float16:
        return False
    d = int(q.size(-1))
    return d in (64, 128)


_OP = False


def _ensure_op():
    global _OP
    if _OP:
        return
    if not (hasattr(torch.ops, "gko") and hasattr(torch.ops.gko, "cutlass_fmha_pair")):

        @torch.library.custom_op("gko::cutlass_fmha_pair", mutates_args=())
        def cutlass_fmha_pair(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_bias: torch.Tensor,
            scale: float,
            causal: int,
            window: int,
            softcap: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            ext = load_ext()
            bias = attn_bias if attn_bias is not None else q.new_empty((0,))
            out, lse = ext.fmha_fwd(
                q, k, v, bias, float(scale), int(causal), int(window), float(softcap)
            )
            return out, lse

        @cutlass_fmha_pair.register_fake
        def _(q, k, v, attn_bias, scale, causal, window, softcap):
            lse = q.new_empty((q.size(0) * q.size(1), q.size(2)), dtype=torch.float32)
            return q.new_empty(q.shape), lse

    from kernels.fused_sdpa import _bwd_pair, _setup_pair

    try:
        torch.library.register_autograd(
            "gko::cutlass_fmha_pair",
            _bwd_pair,
            setup_context=_setup_pair,
        )
    except RuntimeError as e:
        if "already" not in str(e).lower() and "twice" not in str(e).lower():
            raise
    _OP = True


def handwritten_attention(q, k, v, attn_bias, scale, causal=0, window=0, softcap=0.0):
    """CUTLASS-family fused FA as a gko custom op (must show in inductor_custom).

    F.sdpa / flash ATen is NOT a custom kernel: Inductor records efficient_attention
    and leftover inductor_custom stays empty. Prefer gko::cutlass_fmha_pair; on
    load failure call gko::fused_sdpa_pair with the same window.
    """
    squeeze = False
    if q.dim() == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        squeeze = True
    q = q.contiguous() if not q.is_contiguous() else q
    k = k.contiguous() if not k.is_contiguous() else k
    v = v.contiguous() if not v.is_contiguous() else v
    if attn_bias is None:
        attn_bias = q.new_empty((0,))
    elif not attn_bias.is_contiguous() and attn_bias.dim() <= 2:
        attn_bias = attn_bias.contiguous()

    def _triton_pair():
        from kernels.fused_sdpa import _ensure_op as _ensure_triton

        _ensure_triton()
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
        return out

    try:
        if can_handwritten(q):
            _ensure_op()
            out, _lse = torch.ops.gko.cutlass_fmha_pair(
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
    except Exception:
        pass
    out = _triton_pair()
    return out.squeeze(1) if squeeze else out


def addmm_epilogue(bias, mat1, mat2, act=1):
    """CUTLASS tensor-core GEMM + fused epilogue. act: 0=none, 1=gelu, 2=relu."""
    ext = load_ext()
    if bias is None:
        bias = mat1.new_empty((0,))
    return ext.addmm_epilogue(bias, mat1, mat2, int(act))


_SWIGLU_EXT = None
_SWIGLU_FAIL = None
_SWIGLU_SRC = Path(__file__).resolve().parent / "cutlass_fmha" / "gko_cutlass_swiglu.cu"


def load_swiglu_ext():
    """CUTLASS tensor-op GEMM + SwiGLU epilogue extension."""
    global _SWIGLU_EXT, _SWIGLU_FAIL
    if _SWIGLU_EXT is not None:
        return _SWIGLU_EXT
    if _SWIGLU_FAIL is not None:
        raise _SWIGLU_FAIL
    home = cutlass_home()
    if not home:
        _SWIGLU_FAIL = RuntimeError("CUTLASS headers required (GKO_CUTLASS_HOME)")
        raise _SWIGLU_FAIL
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    os.environ.setdefault("CUDA_HOME", cuda_home)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    # Prefer CUDA nvcc + conda ninja; avoid host /usr/local/bin pip-stub ninja.
    os.environ["PATH"] = (
        cuda_home
        + "/bin:/opt/conda/bin:/usr/bin:/bin:"
        + os.environ.get("PATH", "")
    )
    extra = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "-gencode=arch=compute_80,code=sm_80",
        "-I" + str(Path(home) / "include"),
    ]
    try:
        from torch.utils.cpp_extension import load

        _SWIGLU_EXT = load(
            name="gko_cutlass_swiglu_v2",
            sources=[str(_SWIGLU_SRC)],
            extra_cuda_cflags=extra,
            extra_cflags=["-O3", "-std=c++17"],
            extra_include_paths=[str(Path(home) / "include")],
            verbose=True,
            with_cuda=True,
        )
    except Exception as e:
        _SWIGLU_FAIL = e
        raise
    return _SWIGLU_EXT


def linear_swiglu_epilogue(x, weight):
    """CUTLASS tensor-op GEMM + SwiGLU. x (...,D), weight (2H,D) -> (...,H)."""
    ext = load_swiglu_ext()
    return ext.linear_swiglu(x, weight)
