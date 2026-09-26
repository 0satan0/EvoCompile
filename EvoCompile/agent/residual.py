"""Residual observe → depth compose vs breadth jump.

After an eval that *ran*, re-read leftover breaks / fusion holes on the
applied program. If a complementary scheme still matches, compose it onto
the working recipe (depth). If the scheme did not run, or ran but left
nothing complementary, jump to another unused scheme on the *original*
Observation (breadth).

Eval always ``_load``s the uncompiled case and applies the composed
recipe; we never stack OptimizedModules.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Optional

from agent import GKO
from agent.fingerprint import match_context
from agent.state import Observation, SchemeHit

_GENERATE_KERNEL_IDS = {"", "__generate__", "generate"}
_GKO_OP_RE = re.compile(r"gko::([A-Za-z0-9_]+)")
# Recipe kernel id vs the op Inductor actually records. fused_sdpa.py registers
# pair/bwd first; the 12k-char source slice used to miss them.
_KERNEL_OP_ALIASES = {
    "fused_sdpa": frozenset(
        {
            "fused_sdpa",
            "fused_sdpa_pair",
            "fused_sdpa_bwd",
            "fused_attention",
            "cutlass_fmha_pair",
        }
    ),
}


_REWRITE_OPS = {
    "rewrite_class_forward",
    "apply_rewrite",
    "hf_layerdrop_off",
    "allow_logging",
    "disable_forward",
    "disable_method",
}
_KERNEL_OPS = {"replace_pattern"}
_COMPILE_OPS = {"compile", "compile_children"}


def recipe_capabilities(recipe: dict | None) -> dict[str, bool]:
    recipe = recipe or {}
    acts = list(recipe.get("actions") or [])
    ops = {str(a.get("op") or "") for a in acts}
    return {
        "has_rewrite": bool(ops & _REWRITE_OPS),
        "has_layerdrop": "hf_layerdrop_off" in ops,
        "has_kernel": bool(ops & _KERNEL_OPS),
        "has_compile": bool(ops & _COMPILE_OPS) or not acts,
        "has_opt": bool(recipe.get("compile_optimizer")),
        "has_step": bool(recipe.get("compile_step")),
        "has_mark_dynamic": "mark_dynamic" in ops,
        "has_fullgraph": any(
            a.get("op") == "compile" and a.get("fullgraph") for a in acts
        ),
    }


def _action_key(act: dict) -> tuple:
    return (
        str(act.get("op") or ""),
        str(act.get("path") or ""),
        str(act.get("class") or ""),
        str(act.get("kind") or ""),
        str(act.get("kernel") or ""),
        str(act.get("rewrite") or ""),
        str(act.get("name") or ""),
    )


def _rewrites_before_compile(actions: list) -> list:
    """apply_rewrite / class rewrites must run before torch.compile."""
    pre, compile_like = [], []
    for a in actions:
        op = str(a.get("op") or "")
        if op in _COMPILE_OPS:
            compile_like.append(a)
        else:
            pre.append(a)
    return pre + compile_like


def compose_recipes(base: dict | None, extra: dict | None) -> dict:
    """Merge a successful prefix with a complementary scheme's recipe.

    Apply later still hits the original case; this only concatenates the
    closed action list.
    """
    if not base:
        return dict(extra or {})
    if not extra:
        return dict(base)
    out = copy.deepcopy(base)
    extra = dict(extra)
    seen = {_action_key(a) for a in (out.get("actions") or [])}
    merged = list(out.get("actions") or [])
    for act in extra.get("actions") or []:
        k = _action_key(act)
        if k in seen:
            continue
        merged.append(copy.deepcopy(act))
        seen.add(k)
    out["actions"] = _rewrites_before_compile(merged)
    if extra.get("compile_step"):
        # compile(train_step) replaces compile(model); opt lives inside the step.
        out["compile_step"] = True
        if "compile_optimizer" in extra:
            out["compile_optimizer"] = bool(extra.get("compile_optimizer"))
        else:
            out["compile_optimizer"] = False
        out["actions"] = [
            a for a in (out.get("actions") or []) if str(a.get("op") or "") not in _COMPILE_OPS
        ]
    else:
        for flag in ("compile_optimizer", "compile_step"):
            if extra.get(flag) or out.get(flag):
                out[flag] = True
    if extra.get("fullgraph") is True:
        out["fullgraph"] = True
    for bag in ("dynamo", "inductor"):
        if extra.get(bag) or out.get(bag):
            d = dict(out.get(bag) or {})
            d.update(extra.get(bag) or {})
            out[bag] = d
    note_a = str(out.get("note") or "").rstrip(";")
    note_b = str(extra.get("note") or "")
    if note_b and note_b not in note_a:
        out["note"] = f"{note_a}; depth+ {note_b}".strip("; ")
    out.setdefault("model", extra.get("model") or base.get("model"))
    out.setdefault("backend", extra.get("backend") or base.get("backend") or "inductor")
    return out


def patches_only_recipe(recipe: dict) -> dict:
    """Drop compile call-sites so dynamo.explain sees remaining Python breaks."""
    r = copy.deepcopy(recipe or {})
    r["compile_optimizer"] = False
    r["compile_step"] = False
    r["actions"] = [
        a
        for a in (r.get("actions") or [])
        if str(a.get("op") or "") not in _COMPILE_OPS
    ]
    return r


def overlay_observation(obs: Observation, residual: dict | None) -> Observation:
    """Structural fields from obs0; G/I/P from the applied program when present.

    Empty extern/triton lists can mean TraceCapture missed (compile_step).
    Empty inductor_custom is a real observation (identity/S3b has no gko op).
    I-channel leftover is always the wrap this residual came from — the next
    retrieve/repair continues from that compile, not from collect's naive graph.
    """
    feats = dict(obs.features or {})
    # Empty extern/triton can mean TraceCapture missed (compile_step / old cache).
    # Empty inductor_custom is the normal identity/S3b graph — do not keep a
    # previous round's leaked addmm_gelu and treat fusion as already landed.
    skip_empty = {"inductor_extern", "inductor_triton"}
    overlaid = False
    for k, v in (residual or {}).items():
        if v is None:
            continue
        if k in skip_empty and v == []:
            continue
        feats[k] = v
        if k in ("inductor_extern", "inductor_triton", "inductor_custom", "aten_ops"):
            overlaid = True
    if overlaid:
        feats["leftover_source"] = str(
            (residual or {}).get("leftover_source") or "overlay"
        )
    out = Observation(
        model=obs.model,
        suite=obs.suite,
        amp=obs.amp,
        opt=obs.opt,
        features=feats,
        eager=obs.eager,
        vanilla=obs.vanilla,
        model_tree=obs.model_tree,
        forward_src=obs.forward_src,
    )
    return out


def depth_scheme(
    obs: Observation,
    recipe: dict,
    *,
    residual: dict | None = None,
    applied: list[str] | None = None,
) -> Optional[str]:
    """Next complementary scheme from leftover keys, or None if depth is exhausted.

    ``obs`` is the original case (timings stay frozen). ``residual`` overlays
    leftover G/I/P of the applied program when Evaluator captured them.
    """
    blocked = set(applied or [])
    caps = recipe_capabilities(recipe)
    view = overlay_observation(obs, residual)
    ctx = match_context(view)

    ordered: list[tuple[str, bool]] = [
        ("S1", bool(ctx.get("hot_breaks")) and not caps["has_rewrite"]),
        (
            "S2",
            int(ctx.get("layerdrop_n") or 0) >= 1
            and not caps["has_layerdrop"]
            and bool(ctx.get("adam")),
        ),
        (
            "S6-sdpa",
            bool(ctx.get("unfused_bmm_attention"))
            and not caps["has_rewrite"],
        ),
        ("S6-fuse", bool(ctx.get("has_unfused_epilogue")) and not caps["has_kernel"] and "S6-sdpa" not in blocked),
        (
            "S3-step",
            not caps["has_step"]
            and not caps["has_rewrite"]
            and not ctx.get("has_moco")
            and not ctx.get("has_yolo")
            and (
                bool(ctx.get("leftover_step"))
                or bool(ctx.get("has_fnet"))
                or (bool(ctx.get("is_llm")) and bool(ctx.get("zero_break")))
            ),
        ),
        (
            "S3b",
            bool(ctx.get("adam"))
            and bool(ctx.get("zero_break"))
            and not caps["has_opt"]
            and not ctx.get("skinny"),
        ),
        (
            "S5+S3",
            bool(ctx.get("has_lstm")) and bool(ctx.get("adam")) and not caps["has_opt"],
        ),
    ]
    for name, ok in ordered:
        if not ok or name in blocked:
            continue
        return name
    return None


def depth_hit(
    store,
    obs: Observation,
    recipe: dict,
    *,
    residual: dict | None = None,
    applied: list[str] | None = None,
) -> Optional[SchemeHit]:
    name = depth_scheme(obs, recipe, residual=residual, applied=applied)
    if not name:
        return None
    why = (
        f"depth: leftover after {','.join(applied or []) or 'prefix'} "
        f"still matches {name}; compose onto working recipe"
    )
    hit = store.hit_from_scheme(name, obs.model, reason=why, source="depth")
    return hit


def kernel_id_from_recipe(recipe: dict | None) -> str:
    for a in (recipe or {}).get("actions") or []:
        if str(a.get("op") or "") == "replace_pattern":
            return str(a.get("kernel") or "")
    return ""


def _custom_name_list(residual: dict | None) -> list[str]:
    out: list[str] = []
    for it in (residual or {}).get("inductor_custom") or []:
        if isinstance(it, (list, tuple)) and it:
            out.append(str(it[0]).lower())
        elif it:
            out.append(str(it).lower())
    return out


def _kernel_source_paths(kernel_id: str) -> list[Path]:
    root = GKO / "eval" / "kernels"
    return [
        root / f"{kernel_id}.py",
        root / "generated" / f"{kernel_id}.py",
    ]


def expected_fusion_ops(recipe: dict | None) -> set[str]:
    """Op / kernel names this recipe is allowed to count as a fusion land."""
    kid = kernel_id_from_recipe(recipe)
    if kid.lower() in _GENERATE_KERNEL_IDS:
        return set()
    names = {kid.lower()}
    names.update(_KERNEL_OP_ALIASES.get(kid.lower(), ()))
    for path in _kernel_source_paths(kid):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        names.update(m.lower() for m in _GKO_OP_RE.findall(text))
    return names


def fusion_landed(residual: dict | None, recipe: dict | None = None) -> bool:
    """True iff I-channel shows *this recipe's* custom op.

    Leftover ``addmm_gelu`` from a previous process / round must not close
    S3b or a generated kernel that never matched Linear+GELU.
    """
    expected = expected_fusion_ops(recipe)
    if not expected:
        return False
    got = set(_custom_name_list(residual))
    return bool(expected & got)


def fusion_missed(residual: dict | None, recipe: dict | None = None) -> bool:
    """Kernel action ran but did not rewrite the graph."""
    if not recipe_capabilities(recipe)["has_kernel"]:
        return False
    return not fusion_landed(residual, recipe)


def leftover_fusion_hole(obs: Observation, residual: dict | None, recipe: dict | None = None) -> bool:
    from agent.fusion_sites import has_unfused_epilogue, i_channel_for_s6

    view = overlay_observation(obs, residual)
    return has_unfused_epilogue(i_channel_for_s6(view.features)) and not fusion_landed(
        residual, recipe
    )


def observation_for_s6(obs: Observation) -> Observation:
    """View whose I-channel is the wrap S6 continues from (official_step or overlay)."""
    from agent.fusion_sites import i_channel_for_s6, snapshot_i_channel

    feats = obs.features or {}
    src = str(feats.get("leftover_source") or "naive")
    if src in ("overlay", "residual", "official_step"):
        return obs
    s6 = i_channel_for_s6(feats)
    if s6 is feats:
        return obs
    residual = snapshot_i_channel(s6)
    residual["leftover_source"] = "official_step"
    return overlay_observation(obs, residual)


def features_from_trace_summary(summary: dict | None) -> dict[str, Any]:
    summary = summary or {}
    i = summary.get("I") or {}
    return {
        "inductor_extern": i.get("extern_kernels"),
        "inductor_triton": i.get("triton_kernel_names"),
        "inductor_custom": i.get("custom_kernels"),
        "trace_hints": summary.get("hints") or [],
    }


def collect_residual_explain(obs: Observation, recipe: dict) -> dict:
    """G-channel leftover: explain the patched *uncompiled* module."""
    import gc
    import sys

    import torch

    from agent import GKO

    sys.path.insert(0, str(GKO / "eval"))
    from inspect_features import _fwd_explain
    from recipe import apply_input_ops, apply_recipe
    from run_agent_cases import _load, _setup_cwd

    _setup_cwd()
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _bm, model, inputs, opt = _load(obs.model, suite=obs.suite)
    _ = opt
    patched = patches_only_recipe(recipe)
    inputs, _ = apply_input_ops(inputs, patched)
    model, _ = apply_recipe(model, patched)
    return _fwd_explain(model, inputs, obs.amp)
