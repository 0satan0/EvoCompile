"""Fuse silu(x) * y into one pointwise Triton kernel (SwiGLU / SE scale).

Keep vendor GEMM/conv. silu and mul often live in different triton_poi_* names.
"""

import torch

aten = torch.ops.aten

_KERNEL = None
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _silu_mul_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + off, mask=mask, other=0.0).to(tl.float32)
        s = x * (1.0 / (1.0 + tl.exp(-x)))
        tl.store(o_ptr + off, (s * y).to(o_ptr.dtype.element_ty), mask=mask)

    _KERNEL = _silu_mul_kernel
except Exception:
    _KERNEL = None


def _eager(x, y):
    return torch.nn.functional.silu(x) * y


def _triton(x, y):
    if _KERNEL is None or not x.is_cuda or x.shape != y.shape:
        return _eager(x, y)
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _KERNEL[grid](x.view(-1), y.view(-1), out.view(-1), n, BLOCK=BLOCK)
    return out


def _setup_context(ctx, inputs, output):
    x, y = inputs
    ctx.save_for_backward(x, y)


def _backward(ctx, grad_out):
    x, y = ctx.saved_tensors
    sx = torch.nn.functional.silu(x)
    gy = grad_out * sx
    # d silu/dx = silu(x) * (1 + x * (1 - sigmoid(x))) but use autograd formula:
    sig = torch.sigmoid(x)
    gx = grad_out * y * (sig * (1 + x * (1 - sig)))
    return gx, gy


_OP_READY = False
_AUTOGRADED = False
_PASS_INSTALLED = False


def _ensure_op():
    global _OP_READY, _AUTOGRADED
    if not _OP_READY:
        if not (hasattr(torch.ops, "gko") and hasattr(torch.ops.gko, "silu_mul")):

            @torch.library.custom_op("gko::silu_mul", mutates_args=())
            def silu_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return _triton(x, y)

            @silu_mul.register_fake
            def _(x, y):
                return torch.empty_like(x)

        _OP_READY = True
    if not _AUTOGRADED:
        try:
            torch.library.register_autograd(
                "gko::silu_mul", _backward, setup_context=_setup_context
            )
        except RuntimeError as e:
            if "already" not in str(e).lower():
                raise
        _AUTOGRADED = True


def _is_silu(node) -> bool:
    tgt = getattr(node, "target", None)
    if tgt in (aten.silu.default, aten.silu):
        return True
    return getattr(tgt, "__name__", "") == "silu"


def _is_mul(node) -> bool:
    tgt = getattr(node, "target", None)
    return tgt in (aten.mul.Tensor, aten.mul.default, aten.mul)


def fuse_silu_mul(graph) -> int:
    op = torch.ops.gko.silu_mul
    n = 0
    for node in list(graph.nodes):
        if not _is_mul(node) or len(node.args) < 2:
            continue
        a, b = node.args[0], node.args[1]
        silu_n, other = None, None
        if hasattr(a, "op") and _is_silu(a):
            silu_n, other = a, b
        elif hasattr(b, "op") and _is_silu(b):
            silu_n, other = b, a
        if silu_n is None or len(silu_n.users) != 1:
            continue
        x = silu_n.args[0]
        with graph.inserting_before(node):
            new = graph.call_function(op, args=(x, other))
        new.meta = dict(node.meta)
        node.replace_all_uses_with(new)
        graph.erase_node(node)
        if not silu_n.users:
            graph.erase_node(silu_n)
        n += 1
    if n:
        graph.lint()
    return n


def _pass(graph_or_gm) -> None:
    graph = getattr(graph_or_gm, "graph", graph_or_gm)
    fuse_silu_mul(graph)


_pass._gko_graph_native = True  # type: ignore[attr-defined]


def register() -> None:
    global _PASS_INSTALLED
    _ensure_op()
    if _PASS_INSTALLED:
        return
    from kernels.pass_install import install_inductor_pass, install_post_grad_pass

    install_inductor_pass("pre_grad_custom_pass", _pass, key="silu_mul")
    install_post_grad_pass(_pass, key="silu_mul")
    _PASS_INSTALLED = True
