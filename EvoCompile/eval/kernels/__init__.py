"""Kernel catalog: named Triton / custom ops inserted via replace_pattern.

Do not drop kernels into Inductor's cache directory. Each entry is a git-tracked
module under eval/kernels/. Recipe JSON names the id:

  {"op": "replace_pattern", "kernel": "addmm_gelu"}

``register(name)`` installs the custom op (with autograd for training) plus
pre-grad and post-grad FX passes *before* torch.compile. Inductor then sees
one opaque fused op instead of addmm+gelu.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

KERNELS: dict[str, dict] = {
    "addmm_gelu": {
        "module": "kernels.addmm_gelu",
        "match": "gelu(addmm(bias, mat1, mat2)) or gelu(mm + bias)",
        "note": (
            "Missed GEMM epilogue: cuBLAS addmm + separate triton gelu. "
            "Replace with gko.addmm_gelu: CUTLASS tensor-op GEMM+GELU on fp16 "
            "(Triton tl.dot fallback). Same idea as FA leftover: keep vendor-class "
            "GEMM, fuse the pointwise into the epilogue."
        ),
    },
    "gemm_swiglu": {
        "module": "kernels.gemm_swiglu",
        "match": "SwiGLU FFN gate/up: linear to 2H then silu*mul",
        "note": (
            "HARD RULE: any custom kernel whose critical path is GEMM must use "
            "CUTLASS tensor-op (or cuBLAS/cuDNN vendor GEMM + thin epi), NEVER "
            "handwritten float GEMM and NEVER Triton tl.dot as the primary matmul. "
            "True GEMM+SwiGLU epilogue (not standalone silu_mul). Preferred path: "
            "cutlass_fmha.linear_swiglu_epilogue / gko_cutlass_swiglu.cu. "
            "Triton tl.dot / gko_linear_swiglu.cu are demoted (A100 microbench ~0.05–0.7× "
            "vs cuBLAS). Prefer over opaque silu_and_mul."
        ),
    },
    "fused_sdpa": {
        "module": "kernels.fused_sdpa",
        "match": "unfused bmm + softmax + bmm attention (no in-tree Flash)",
        "note": (
            "Flash-style tiled online-softmax Triton OR vendor CUTLASS FMHA. "
            "register(backend='auto') picks Triton for 4D rel-bias/softcap/small-S, "
            "CUTLASS vendor FMHA for long causal ≤16 layers / longformer / n_layer 30–40. "
            "Must hook attention modules (apply_rewrite fused_sdpa_attn) then compile_step."
        ),
    },
    "silu_mul": {
        "module": "kernels.silu_mul",
        "match": "silu(x) * y split across two triton kernels (SwiGLU / SE)",
        "note": (
            "One gko.silu_mul pointwise kernel; keep vendor GEMM/conv. "
            "Do NOT use for Titan dense Llama — prefer gemm_swiglu or stock silu."
        ),
    },
}

_installed: set[str] = set()
_ATTN_CALL = None
GENERATED_DIR = Path(__file__).resolve().parent / "generated"

_GKO_OP_RE = re.compile(r"""custom_op\(\s*['"]gko::([A-Za-z0-9_]+)""")


def attn_call():
    """Forward used by fused_sdpa_attn: this recipe's torch.ops.gko.* (or catalog FA)."""
    return _ATTN_CALL


def _wrap_gko_attn(op):
    """Call a generated gko op with the attention-hook layout; unwrap (out, lse)."""
    schema = str(getattr(op, "_schema", "") or op)
    try:
        overloads = list(op.overloads()) if hasattr(op, "overloads") else []
        if overloads:
            ov = getattr(op, overloads[0])
            schema = str(getattr(ov, "_schema", schema))
    except Exception:
        pass
    inner = ""
    m = re.search(r"\((.*)\)\s*->", schema)
    if m:
        inner = m.group(1).strip()
    arity = inner.count(",") + 1 if inner else 7

    def fn(q, k, v, attn_bias, scale, causal=0, window=0, softcap=0.0):
        args = [
            q,
            k,
            v,
            attn_bias,
            float(scale),
            int(causal),
            int(window or 0),
            float(softcap or 0.0),
        ]
        out = op(*args[: max(arity, 4)])
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    fn.__name__ = "gko_generated_attn"
    return fn


def _first_gko_fwd_op(mod):
    import torch

    gko = getattr(torch.ops, "gko", None)
    if gko is None:
        return None
    names = []
    path = getattr(mod, "__file__", "") or ""
    if path:
        try:
            text = Path(path).read_text()
        except Exception:
            text = ""
        names = _GKO_OP_RE.findall(text)
    names = [n for n in names if not n.endswith("_bwd") and "backward" not in n.lower()]
    if not names:
        return None
    pick = names[0]
    for n in names:
        nl = n.lower()
        if any(t in nl for t in ("pair", "attn", "sdpa", "fa", "attention", "fmha")):
            pick = n
            break
    return getattr(gko, pick, None)


def _bind_attn_from_mod(mod, name: str) -> None:
    global _ATTN_CALL
    fn = getattr(mod, "fused_attention", None)
    if callable(fn):
        _ATTN_CALL = fn
        return
    op = _first_gko_fwd_op(mod)
    if op is not None:
        _ATTN_CALL = _wrap_gko_attn(op)


def reset_installed() -> None:
    """Forget catalog/generated registers and uninstall inductor FX passes.

    ``register()`` is process-level. Without this, a later identity/S3b eval
    still rewrites Linear+GELU and I-channel residual lies to the loop.
    """
    global _ATTN_CALL
    import sys

    _ATTN_CALL = None
    _installed.clear()
    for _name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        file = str(getattr(mod, "__file__", "") or "").replace("\\", "/")
        is_kernel = (
            _name in ("kernels.addmm_gelu", "kernels.fused_sdpa", "kernels.silu_mul")
            or _name.startswith("kernels.generated.")
            or "/eval/kernels/" in file
        )
        if not is_kernel or not hasattr(mod, "_PASS_INSTALLED"):
            continue
        try:
            setattr(mod, "_PASS_INSTALLED", False)
        except Exception:
            pass
    from kernels.pass_install import reset_passes

    reset_passes()


def list_kernels() -> list[str]:
    names = set(KERNELS)
    if GENERATED_DIR.is_dir():
        names.update(p.stem for p in GENERATED_DIR.glob("*.py") if p.stem != "__init__")
    return sorted(names)


def register(name: str, backend: str = "") -> str:
    if name in _installed:
        return f"replace_pattern:{name} (already registered)"
    spec = KERNELS.get(name)
    if spec:
        from importlib import import_module

        mod = import_module(spec["module"])
        fn = getattr(mod, "register")
        if backend:
            try:
                fn(backend)
            except TypeError:
                fn()
        else:
            fn()
        _installed.add(name)
        if name == "fused_sdpa":
            _bind_attn_from_mod(mod, name)
        return f"replace_pattern:{name}" + (f" backend={backend}" if backend else "")
    gen = GENERATED_DIR / f"{name}.py"
    if gen.is_file():
        import importlib.util

        spec_mod = importlib.util.spec_from_file_location(f"kernels.generated.{name}", gen)
        if spec_mod is None or spec_mod.loader is None:
            raise ValueError(f"cannot load generated kernel {name}")
        mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(mod)
        from kernels.pass_install import install_generated_pass, normalize_pre_pass

        try:
            getattr(mod, "register")()
        except AttributeError as e:
            msg = str(e)
            if "does not exist" in msg and ("_inductor.config" in msg or "_gko_" in msg):
                install_generated_pass(mod, name)
            else:
                raise
        except TypeError as e:
            msg = str(e)
            if "install_inductor_pass" in msg or "pass_fn" in msg:
                install_generated_pass(mod, name)
            else:
                raise
        normalize_pre_pass()
        _installed.add(name)
        _bind_attn_from_mod(mod, name)
        return f"replace_pattern:{name} (generated)"
    raise ValueError(f"unknown kernel {name!r}; catalog={list_kernels()}")
