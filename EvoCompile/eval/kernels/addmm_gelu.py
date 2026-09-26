"""Fused addmm + GELU: first replace_pattern kernel.

Vanilla Inductor often lowers Linear+GELU as ``extern_kernels.addmm`` (cuBLAS)
plus a separate ``triton_per_fused_gelu_*`` pointwise. That is a missed GEMM
epilogue. This module:

1. Registers opaque ``gko::addmm_gelu`` (Triton on CUDA, aten fallback else)
   with a traceable autograd formula so training graphs can keep the fused op.
2. Installs a *pre-grad* FX pass (and post-grad for inference leftovers) that
   rewrites ``gelu(addmm/linear)`` to that op *before* AOT decomposes backward.
   Insertion is in-process — not a patch of output_code.py.
"""

import torch

aten = torch.ops.aten

_KERNEL = None
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _addmm_gelu_kernel(
        a_ptr,
        b_ptr,
        bias_ptr,
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
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_off = k * BLOCK_K
            k_mask = (offs_k + k_off) < K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=k_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(
            c_ptrs,
            acc.to(c_ptr.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    _KERNEL = _addmm_gelu_kernel
except Exception:
    _KERNEL = None


def _eager_addmm_gelu(bias, mat1, mat2):
    if mat1.dim() == 2 and mat2.dim() == 2:
        return torch.nn.functional.gelu(torch.addmm(bias, mat1, mat2), approximate="none")
    return torch.nn.functional.gelu(torch.matmul(mat1, mat2) + bias, approximate="none")


def _triton_addmm_gelu(bias, mat1, mat2):
    if _KERNEL is None or not mat1.is_cuda:
        return _eager_addmm_gelu(bias, mat1, mat2)
    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    bias = bias.contiguous()
    M, K = mat1.shape
    K2, N = mat2.shape
    if K != K2:
        raise ValueError(f"addmm_gelu K mismatch {K} vs {K2}")
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _KERNEL[grid](
        mat1,
        mat2,
        bias,
        out,
        M,
        N,
        K,
        mat1.stride(0),
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


def _cutlass_addmm_gelu(bias, mat1, mat2):
    """Vendor-class GEMM (CUTLASS tensor-op) + GELU in the epilogue.

    Catalog Triton ``tl.dot`` loses to cuBLAS on these shapes. Fusing GELU into
    a CUTLASS GEMM keeps the tensor-core mainloop and drops the extra gelu
    launch that official leftover still pays (``extern addmm`` + triton gelu).
    """
    import os

    if os.environ.get("GKO_ADDMM_BACKEND", "cutlass").strip().lower() == "triton":
        return None
    if mat1 is None or mat2 is None or not mat1.is_cuda:
        return None
    if mat1.dtype != torch.float16 or mat2.dtype != torch.float16:
        return None
    if mat2.dim() != 2:
        return None
    k = int(mat1.shape[-1])
    n = int(mat2.shape[-1])
    if k % 8 or n % 8 or int(mat2.shape[0]) != k:
        return None
    try:
        from kernels.cutlass_gemm_attn import addmm_epilogue

        return addmm_epilogue(bias, mat1, mat2, act=1)
    except Exception:
        return None


def _addmm_gelu_forward(bias, mat1, mat2):
    """addmm(bias, mat1, mat2) then erf-GELU. mat1 may be [..., K] (Linear)."""
    fused = _cutlass_addmm_gelu(bias, mat1, mat2)
    if fused is not None:
        return fused
    if mat2.dim() != 2:
        return _eager_addmm_gelu(bias, mat1, mat2)
    if mat1.dim() != 2:
        lead = mat1.shape[:-1]
        out = _triton_addmm_gelu(bias, mat1.reshape(-1, mat1.shape[-1]), mat2)
        return out.reshape(lead + (mat2.shape[1],))
    return _triton_addmm_gelu(bias, mat1, mat2)


def _addmm_pre_gelu(bias, mat1, mat2):
    if mat1.dim() == 2 and mat2.dim() == 2:
        return torch.addmm(bias, mat1, mat2)
    return torch.matmul(mat1, mat2) + bias


def _addmm_gelu_setup_context(ctx, inputs, output):
    bias, mat1, mat2 = inputs
    ctx.save_for_backward(bias, mat1, mat2)


def _addmm_gelu_backward(ctx, grad_output):
    """Traceable: recompute pre-GELU, then gelu_backward + Linear-style grads."""
    bias, mat1, mat2 = ctx.saved_tensors
    z = _addmm_pre_gelu(bias, mat1, mat2)
    gz = torch.ops.aten.gelu_backward.default(grad_output, z, approximate="none")
    needs = getattr(ctx, "needs_input_grad", None) or (True, True, True)
    gz2 = gz.reshape(-1, gz.shape[-1])
    gbias = gz2.sum(0) if needs[0] else None
    gmat1 = gz2.mm(mat2.t()).reshape(mat1.shape) if needs[1] else None
    gmat2 = mat1.reshape(-1, mat1.shape[-1]).t().mm(gz2) if needs[2] else None
    return gbias, gmat1, gmat2


_OP_READY = False
_AUTOGRADED = False


def _ensure_op():
    global _OP_READY, _AUTOGRADED
    if not _OP_READY:
        if not (hasattr(torch.ops, "gko") and hasattr(torch.ops.gko, "addmm_gelu")):

            @torch.library.custom_op("gko::addmm_gelu", mutates_args=())
            def addmm_gelu(
                bias: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
            ) -> torch.Tensor:
                return _addmm_gelu_forward(bias, mat1, mat2)

            @addmm_gelu.register_fake
            def _(bias, mat1, mat2):
                n = mat2.shape[-1]
                return torch.empty(
                    mat1.shape[:-1] + (n,),
                    device=mat1.device,
                    dtype=mat1.dtype,
                )

        _OP_READY = True
    if not _AUTOGRADED:
        try:
            torch.library.register_autograd(
                "gko::addmm_gelu",
                _addmm_gelu_backward,
                setup_context=_addmm_gelu_setup_context,
            )
        except RuntimeError as e:
            if "already" not in str(e).lower():
                raise
        _AUTOGRADED = True


def _owned_module(node):
    if getattr(node, "op", None) != "call_module":
        return None
    gm = getattr(getattr(node, "graph", None), "owning_module", None)
    if gm is None:
        return None
    tgt = str(node.target)
    try:
        return gm.get_submodule(tgt)
    except Exception:
        return dict(gm.named_modules()).get(tgt)


def _is_gelu(node) -> bool:
    if getattr(node, "op", None) == "call_module":
        mod = _owned_module(node)
        if isinstance(mod, torch.nn.GELU):
            return getattr(mod, "approximate", "none") in (None, "none")
        return False
    tgt = node.target
    if tgt not in (aten.gelu.default, aten.gelu) and not (
        getattr(tgt, "__name__", "") == "gelu"
        and str(getattr(tgt, "__module__", "")).startswith("torch")
    ):
        return False
    approx = (node.kwargs or {}).get("approximate", "none")
    return approx in (None, "none")


def _is_linear(node) -> bool:
    tgt = getattr(node, "target", None)
    if tgt in (aten.linear.default, aten.linear, torch.nn.functional.linear):
        return True
    if tgt is getattr(torch._C._nn, "linear", None):
        return True
    name = getattr(tgt, "__name__", "")
    mod = str(getattr(tgt, "__module__", "") or "")
    return name == "linear" and ("torch" in mod or "aten" in mod)


def _is_linear_module(node) -> bool:
    mod = _owned_module(node)
    return isinstance(mod, torch.nn.Linear) and mod.bias is not None


def _linear_operands(node):
    args = node.args or ()
    kwargs = node.kwargs or {}
    inp = args[0] if args else kwargs.get("input")
    weight = args[1] if len(args) > 1 else kwargs.get("weight")
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    return inp, weight, bias


def _is_addmm(node) -> bool:
    tgt = getattr(node, "target", None)
    if tgt in (aten.addmm.default, aten.addmm, torch.addmm):
        return True
    return getattr(tgt, "__name__", "") == "addmm"


def _is_mm(node) -> bool:
    tgt = getattr(node, "target", None)
    return tgt in (aten.mm.default, aten.mm)


def _is_add(node) -> bool:
    tgt = getattr(node, "target", None)
    return tgt in (aten.add.Tensor, aten.add.default, aten.add)


def _is_reshape(node) -> bool:
    tgt = getattr(node, "target", None)
    names = {aten.view.default, aten.reshape.default}
    unsafe = getattr(getattr(aten, "_unsafe_view", None), "default", None)
    if unsafe is not None:
        names.add(unsafe)
    return tgt in names


def _is_mul(node) -> bool:
    tgt = getattr(node, "target", None)
    return tgt in (aten.mul.Tensor, aten.mul.default, aten.mul)


def _is_erf(node) -> bool:
    tgt = getattr(node, "target", None)
    return tgt in (aten.erf.default, aten.erf)


def _as_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _other_operand(node, tensor_node):
    if not node.args or len(node.args) < 2:
        return None
    a, b = node.args[0], node.args[1]
    if a is tensor_node:
        return b
    if b is tensor_node:
        return a
    return None


def _mul_by(tensor_node, value: float, tol: float = 1e-6):
    for user in list(tensor_node.users):
        if not _is_mul(user):
            continue
        other = _other_operand(user, tensor_node)
        f = _as_float(other)
        if f is not None and abs(f - value) < tol:
            return user
    return None


def _add_const(tensor_node, value: float, tol: float = 1e-6):
    for user in list(tensor_node.users):
        if not _is_add(user):
            continue
        other = _other_operand(user, tensor_node)
        f = _as_float(other)
        if f is not None and abs(f - value) < tol:
            return user
    return None


def _gelu_chain_from_addmm(addmm_node):
    """Match decomposed gelu: 0.5 * x * (1 + erf(x / sqrt(2)))."""
    mul_half = _mul_by(addmm_node, 0.5)
    mul_scale = _mul_by(addmm_node, 0.7071067811865476)
    if mul_half is None or mul_scale is None:
        return None
    erf_users = [u for u in mul_scale.users if _is_erf(u)]
    if len(erf_users) != 1:
        return None
    erf_n = erf_users[0]
    add1 = _add_const(erf_n, 1.0)
    if add1 is None:
        return None
    # final: mul(half, 1+erf)
    for user in add1.users:
        if _is_mul(user) and (user.args[0] is mul_half or user.args[1] is mul_half):
            return {
                "out": user,
                "erase": [user, add1, erf_n, mul_scale, mul_half],
            }
    return None


def _addmm_from_mm_add(src):
    """If src is add(mm, bias) return (bias, mat1, mat2, extra_erase)."""
    if _is_add(src) and len(src.args) >= 2:
        a, b = src.args[0], src.args[1]
        if hasattr(a, "op") and _is_mm(a) and len(a.users) == 1:
            return b, a.args[0], a.args[1], [src, a]
        if hasattr(b, "op") and _is_mm(b) and len(b.users) == 1:
            return a, b.args[0], b.args[1], [src, b]
    return None


def _reshape_between(src):
    """Peel one or two view/reshape nodes sitting on a GEMM result."""
    wraps = []
    cur = src
    while (
        cur is not None
        and hasattr(cur, "op")
        and _is_reshape(cur)
        and len(cur.args) >= 1
        and len(wraps) < 2
    ):
        wraps.append(cur)
        cur = cur.args[0]
    return cur, wraps


def fuse_addmm_gelu(graph: torch.fx.Graph) -> int:
    """Rewrite gelu(addmm/linear) / decomposed-gelu(addmm) → gko.addmm_gelu.

    Pre-grad: addmm/linear's only forward user is the epilogue (dropout sits on GELU).
    Autograd of the custom op owns backward, so we do not wait for post-grad.
    Post-grad still skips when addmm has leftover users (already-decomposed bwd).
    """
    _ensure_op()
    op = torch.ops.gko.addmm_gelu.default
    n = 0
    for node in list(graph.nodes):
        bias = mat1 = mat2 = None
        extra_erase = []
        reshape_wrap = []
        addmm_node = None
        gelu_out = None
        erase = []
        weight_t = False
        from_module = False

        if _is_gelu(node):
            src = node.args[0] if node.args else None
            if src is None or not hasattr(src, "op"):
                continue
            src, reshape_wrap = _reshape_between(src)
            if _is_addmm(src):
                addmm_node = src
                bias, mat1, mat2 = src.args[0], src.args[1], src.args[2]
            elif _is_linear(src):
                mat1, weight, bias = _linear_operands(src)
                if mat1 is None or weight is None or bias is None:
                    continue
                addmm_node = src
                mat2 = weight
                weight_t = True
            elif _is_linear_module(src):
                addmm_node = src
                mat1 = src.args[0]
                from_module = True
                weight_t = True
            else:
                parsed = _addmm_from_mm_add(src)
                if parsed is None:
                    continue
                bias, mat1, mat2, extra_erase = parsed
                addmm_node = extra_erase[1]
            gelu_out = node
            erase = [node]
        elif node.op == "call_function" and _is_addmm(node):
            chain = _gelu_chain_from_addmm(node)
            if chain is None:
                rusers = [u for u in node.users if _is_reshape(u)]
                if len(rusers) == 1:
                    chain = _gelu_chain_from_addmm(rusers[0])
                    if chain is not None:
                        reshape_wrap = [rusers[0]]
            if chain is None:
                continue
            addmm_node = node
            bias, mat1, mat2 = node.args[0], node.args[1], node.args[2]
            gelu_out = chain["out"]
            erase = chain["erase"]
        else:
            continue

        if addmm_node is None or gelu_out is None:
            continue
        extra_erase = list(reshape_wrap) + list(extra_erase)
        gelu_users = set(erase) | set(extra_erase)
        leftover = [u for u in addmm_node.users if u not in gelu_users]
        if leftover:
            continue
        skip_view = False
        for v in reshape_wrap:
            if [u for u in v.users if u not in gelu_users]:
                skip_view = True
                break
        if skip_view:
            continue

        with graph.inserting_before(gelu_out):
            if from_module:
                w = graph.get_attr(str(addmm_node.target) + ".weight")
                bias = graph.get_attr(str(addmm_node.target) + ".bias")
                mat2 = graph.call_function(aten.t.default, args=(w,))
                new = graph.call_function(op, args=(bias, mat1, mat2))
            else:
                if weight_t:
                    mat2 = graph.call_function(aten.t.default, args=(mat2,))
                new = graph.call_function(op, args=(bias, mat1, mat2))
            for v in reversed(reshape_wrap):
                new = graph.call_function(
                    v.target,
                    args=(new,) + tuple(v.args[1:]),
                    kwargs=dict(v.kwargs or {}),
                )
        new.meta = dict(gelu_out.meta)
        gelu_out.replace_all_uses_with(new)
        for dead in erase + extra_erase + [addmm_node]:
            if dead.op != "placeholder" and not dead.users:
                try:
                    graph.erase_node(dead)
                except Exception:
                    pass
        n += 1
    if n:
        graph.lint()
    return n


def _addmm_gelu_pass(graph_or_gm) -> None:
    """Inductor 2.4 passes a Graph; newer builds may pass a GraphModule."""
    graph = getattr(graph_or_gm, "graph", graph_or_gm)
    fuse_addmm_gelu(graph)


_addmm_gelu_pass._gko_graph_native = True  # type: ignore[attr-defined]


_PASS_INSTALLED = False


def register() -> None:
    """Install custom op + pre-grad (training) and post-grad (inference) passes."""
    global _PASS_INSTALLED
    _ensure_op()
    if _PASS_INSTALLED:
        return
    from kernels.pass_install import install_inductor_pass, install_post_grad_pass

    install_inductor_pass("pre_grad_custom_pass", _addmm_gelu_pass, key="addmm_gelu")
    install_post_grad_pass(_addmm_gelu_pass, key="addmm_gelu")
    _PASS_INSTALLED = True
