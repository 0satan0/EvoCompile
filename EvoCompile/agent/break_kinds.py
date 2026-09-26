"""Classify dynamo.explain break_sample strings into a closed taxonomy.

The agent does not rewrite user ``forward`` source. These tags pick a whitelist
recipe op (allow_logging, rewrite_class_forward kind=..., skip_code, ...).
"""

from __future__ import annotations

from typing import Any, Iterable

# Closed kinds. Retrieval predicates and RECIPE_SCHEMA both use these names.
KINDS = ("side_effect", "data_dependent", "vendor", "setattr")

_SIDE = (
    "print(",
    "print ",
    "builtin print",
    "builtins.print",
    "logging.",
    "logger.",
    "warnings.",
    "sys.stdout",
    "stdout",
    "side effect",
    "side-effect",
    "reorderable_logging",
)
_SETATTR = (
    "setattr",
    "store_attr",
    "__setattr__",
    "create_grids",
    "problem_type",
)
_DATA = (
    ".item()",
    "item()",
    "nonzero",
    "tolist",
    "_local_scalar_dense",
    "datadependent",
    "data-dependent",
    "data dependent",
    "dynamic slicing",
    "unbacked",
)
_VENDOR = (
    "lstm",
    "rnn",
    "gru",
    "cudnn",
    "fakequant",
    "fake_quant",
    "embeddingbag",
    "embedding_bag",
)
_DYNAMIC = ("dynamic", "dynamic shapes", "not a constant")


def _blob(sample: Any) -> str:
    return str(sample or "").lower()


def classify_one(sample: Any) -> set[str]:
    s = _blob(sample)
    out: set[str] = set()
    if any(t in s for t in _SIDE):
        out.add("side_effect")
    if any(t in s for t in _SETATTR):
        out.add("setattr")
    if any(t in s for t in _DATA):
        out.add("data_dependent")
    if any(t in s for t in _VENDOR):
        out.add("vendor")
    return out


def classify_breaks(samples: Iterable[Any] | None) -> dict:
    """Flags + kind list from dynamo.explain break_reasons (or similar strings)."""
    rows = list(samples or [])
    kinds: set[str] = set()
    for row in rows:
        kinds |= classify_one(row)
    blob = " ".join(_blob(x) for x in rows)
    return {
        "break_kinds": sorted(kinds),
        "has_side_effect": "side_effect" in kinds,
        "has_print_break": "side_effect" in kinds,
        "has_vendor_break": "vendor" in kinds,
        "has_setattr": "setattr" in kinds or "problem_type" in blob,
        "has_nonzero": "nonzero" in blob,
        "has_item": "item" in blob,
        "has_dynamic": any(t in blob for t in _DYNAMIC) or "data_dependent" in kinds,
        "has_lstm": "lstm" in blob,
        "has_rnn": "rnn" in blob,
        "has_embedbag": "embeddingbag" in blob or "embedding_bag" in blob,
    }


def mark_dynamic_action(ctx: dict) -> dict | None:
    """Closed mark_dynamic action from input layout. Never emit a dict key on a tuple."""
    if not ctx.get("needs_mark_dynamic"):
        return None
    kind = str(ctx.get("input_kind") or "")
    ndim = ctx.get("input_ndim")
    try:
        ndim = int(ndim) if ndim is not None else 0
    except (TypeError, ValueError):
        ndim = 0
    name = str(ctx.get("name") or "").lower()
    lin = float(ctx.get("lin") or 0.0)
    transformerish = lin >= 0.5 or any(
        t in name for t in ("bert", "gpt", "t5", "llama", "bart", "whisper")
    )
    if kind == "dict":
        keys = list(ctx.get("input_keys") or [])
        key = "input_ids" if ctx.get("has_input_ids") or "input_ids" in keys else (
            keys[0] if keys else None
        )
        if not key:
            return None
        dim = 1 if ndim >= 2 else 0
        return {"op": "mark_dynamic", "input": str(key), "dim": dim}
    dim = 1 if transformerish and ndim >= 2 else 0
    return {"op": "mark_dynamic", "index": 0, "dim": dim}


def attach_mark_dynamic(recipe: dict, ctx: dict) -> dict:
    """Prepend mark_dynamic onto a compile recipe. Skip eager/skip_compile."""
    if not recipe:
        return recipe
    actions = list(recipe.get("actions") or [])
    if not actions:
        return recipe
    if any(a.get("op") == "mark_dynamic" for a in actions):
        return recipe
    act = mark_dynamic_action(ctx)
    if act is None:
        return recipe
    out = dict(recipe)
    out["actions"] = [act] + actions
    return out
