"""Install inductor post_grad_custom_pre_pass without unknown config attrs.

Torch 2.4 ConfigModule rejects setattr of fields that are not in the schema:

    AttributeError: torch._inductor.config._gko_* does not exist

Idempotency lives here (process-level set), not on inductor.config.
The pass callable may take a Graph (2.4 post_grad) or a GraphModule (LLM).
"""

from __future__ import annotations

from typing import Any, Callable

_INSTALLED: set[str] = set()


def as_graph_pass(fn: Callable) -> Callable:
    """Accept Graph or GraphModule. Inductor 2.4 passes a Graph."""
    if getattr(fn, "_gko_graph_wrapped", False):
        return fn
    if getattr(fn, "_gko_graph_native", False):
        return fn
    if hasattr(fn, "uuid"):
        return fn

    def _call(graph_or_gm: Any) -> Any:
        if hasattr(graph_or_gm, "graph") and callable(getattr(graph_or_gm, "recompile", None)):
            return fn(graph_or_gm)

        class _GMShim:
            def __init__(self, graph: Any) -> None:
                self.graph = graph

            def recompile(self) -> None:
                lint = getattr(self.graph, "lint", None)
                if callable(lint):
                    lint()

        return fn(_GMShim(graph_or_gm))

    _call.__name__ = getattr(fn, "__name__", "gko_pre_pass")
    _call._gko_graph_wrapped = True  # type: ignore[attr-defined]
    return _call


def ensure_graph_pass(fn: Any) -> Any:
    if fn is None:
        return None
    if isinstance(fn, (list, tuple)):
        parts = [ensure_graph_pass(x) for x in fn]
        return _compose_callables(parts)
    return as_graph_pass(fn)


def _compose_callables(fns: list) -> Callable:
    """Torch 2.4 calls post_grad_custom_pre_pass(graph) — a list is not callable."""
    fns = [f for f in fns if f is not None]

    def _call(graph: Any) -> None:
        for fn in fns:
            if callable(fn):
                fn(graph)

    _call._gko_graph_native = True  # type: ignore[attr-defined]
    _call.__name__ = "gko_composed_pre_pass"
    return _call


def install_inductor_pass(attr: str, pass_fn: Callable, *, key: str = "") -> None:
    import torch._inductor.config as inductor_config

    if not callable(pass_fn):
        raise TypeError(
            "install_inductor_pass(attr, pass_fn): pass_fn must be a callable, "
            f"got {type(pass_fn).__name__}"
        )
    if not hasattr(inductor_config, attr):
        return
    token = f"{attr}:{key or getattr(pass_fn, '__name__', '') or id(pass_fn)}"
    if token in _INSTALLED:
        return
    wrapped = ensure_graph_pass(pass_fn)
    existing = getattr(inductor_config, attr)
    if existing is None:
        setattr(inductor_config, attr, wrapped)
    else:
        setattr(inductor_config, attr, _compose_callables([existing, wrapped]))
    _INSTALLED.add(token)


def install_post_grad_pass(pass_fn: Callable, *, key: str = "") -> None:
    install_inductor_pass("post_grad_custom_pre_pass", pass_fn, key=key)


def normalize_pre_pass() -> None:
    import torch._inductor.config as inductor_config

    cur = inductor_config.post_grad_custom_pre_pass
    if cur is None:
        return
    inductor_config.post_grad_custom_pre_pass = ensure_graph_pass(cur)


def reset_passes() -> None:
    """Drop process-level FX passes so the next eval only sees this recipe."""
    _INSTALLED.clear()
    try:
        import torch._inductor.config as inductor_config
    except Exception:
        return
    for attr in ("pre_grad_custom_pass", "post_grad_custom_pre_pass"):
        if hasattr(inductor_config, attr):
            setattr(inductor_config, attr, None)


def install_generated_pass(mod: Any, name: str) -> None:
    """Recover when generated register() crashed on unknown config attrs."""
    fn = getattr(mod, "_post_grad_pass", None)
    if fn is None:
        for attr in dir(mod):
            if attr in ("_PASS_INSTALLED", "register"):
                continue
            obj = getattr(mod, attr, None)
            if not callable(obj):
                continue
            al = attr.lower()
            if attr.startswith("_") and (
                attr.endswith("_pass") or "rewrite" in al or "fx_pass" in al or al.endswith("_pattern")
            ):
                fn = obj
                break
    if fn is None:
        raise AttributeError(f"generated kernel {name} has no post-grad pass to install")
    install_inductor_pass("pre_grad_custom_pass", fn, key=name)
    install_post_grad_pass(fn, key=name)
