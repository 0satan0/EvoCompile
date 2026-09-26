"""Repair LLM: traceback + failed recipe → new recipe. No scheme retrieval.

Repair fixes THIS attempt so the intended strategy lands (runs + matches
eager). Dispatch is per intervention layer (A region / B method / C kernel).
It does not revert to an earlier best recipe — the orchestrator already
keeps that as the run result and as the next optimize base.
"""

from __future__ import annotations

from typing import Optional

from agent.llm import LLMClient, parse_json_object
from agent.prompts import system_repair, user_repair
from agent.state import Observation

_LAYER_B_OPS = frozenset(
    {
        "apply_rewrite",
        "rewrite_class_forward",
        "hf_layerdrop_off",
        "speech_vectorize_pad_mask",
        "skip_code_on_types",
    }
)
_CATALOG_KERNELS = frozenset({"fused_sdpa", "addmm_gelu", "silu_mul"})
_CATALOG_HOOKS = frozenset({"fused_sdpa_attn", "sdpa_rewrite"})
_INVENTED_KEYS = frozenset({"fp32_bias"})
_REWRITE_KEEP = frozenset({"op", "rewrite"})


def failed_intervention_layer(recipe: dict | None, error: str = "") -> str:
    """Which layer's operator actually failed. Traceback location wins.

    Prefer Layer C when a generated custom op / FX pass is in the stack, even if
    the call arrived through fused_sdpa_attn (that hook is only the insert path).
    """
    err = error or ""
    el = err.lower()
    acts = list((recipe or {}).get("actions") or [])
    ops = {str(a.get("op") or "") for a in acts}
    has_c = "replace_pattern" in ops
    has_b = bool(ops & _LAYER_B_OPS)
    hook_hit = "fused_sdpa_attn.py" in err or "rewrites/fused_sdpa" in el
    gen_rw = "rewrites/generated" in el or "rewrites.generated" in el
    kernel_hit = any(
        m in el
        for m in (
            "kernels/generated",
            "kernels.generated",
            "torch.ops.gko",
            "gko::",
            "compilationerror",
            "triton.compiler",
            "tensors returned from custom ops",
            "alias any inputs",
            "offending op:",
            "may not alias",
            "pass_install",
            "_gmshim",
            "install_inductor_pass",
            "'str' object is not callable",
        )
    )
    # Generated op / FX pass crashed while hooked: stay on C, not B.
    if kernel_hit and has_c:
        return "C"
    if hook_hit or gen_rw:
        return "B" if has_b else ("C" if has_c else "A")
    if "fail:rel_loss" in el or "numerics" in el:
        if has_c:
            return "C"
        if has_b:
            return "B"
        return "A"
    if has_c:
        return "C"
    if has_b:
        return "B"
    return "A"


def _drop_fullgraph(recipe: dict) -> dict:
    out = dict(recipe)
    out["fullgraph"] = False
    actions = []
    for act in list(out.get("actions") or []):
        a = dict(act)
        if a.get("op") == "compile" and a.get("fullgraph"):
            a["fullgraph"] = False
        actions.append(a)
    out["actions"] = actions
    out["compile_step"] = False
    note = str(out.get("note") or "")
    out["note"] = (note + "; heuristic repair: drop fullgraph").strip("; ")
    return out


def _keep_attempt(recipe: dict, note: str) -> dict:
    out = dict(recipe or {})
    if not isinstance(out.get("actions"), list):
        out["actions"] = list(out.get("actions") or [])
    prev = str(out.get("note") or "")
    out["note"] = (prev + "; " + note).strip("; ")
    return out


_LANDING_OPS = ("replace_pattern", "apply_rewrite")


def _norm_kernel(k: str) -> str:
    k = str(k or "")
    if k.lower() in _CATALOG_KERNELS or k in ("", "generate", "__generate__"):
        return "__generate__"
    return k


def _action_sig(act: dict) -> tuple:
    return (
        str(act.get("op") or ""),
        _norm_kernel(str(act.get("kernel") or "")),
        str(act.get("rewrite") or ""),
    )


def _landing_ops(recipe: dict | None) -> list[dict]:
    out = []
    for a in (recipe or {}).get("actions") or []:
        if str(a.get("op") or "") in _LANDING_OPS:
            out.append(dict(a))
    return out


def strip_invented_action_keys(recipe: dict | None) -> dict:
    """Drop LLM-invented fields (fp32_bias, backend=math) and duplicate inserts."""
    out = dict(recipe or {})
    seen_rp = False
    acts = []
    for raw in list(out.get("actions") or []):
        a = {k: v for k, v in dict(raw).items() if k not in _INVENTED_KEYS}
        op = str(a.get("op") or "")
        if op == "replace_pattern":
            a = {"op": "replace_pattern", "kernel": _norm_kernel(str(a.get("kernel") or ""))}
            if seen_rp:
                continue
            seen_rp = True
        elif op == "apply_rewrite":
            a = {k: a[k] for k in _REWRITE_KEEP if k in a}
            a["op"] = "apply_rewrite"
        acts.append(a)
    out["actions"] = acts
    spec = out.get("kernel_spec")
    if isinstance(spec, dict):
        spec = dict(spec)
        spec["catalog_hint"] = None
        out["kernel_spec"] = spec
    return out


def preserve_kernel_landing(failed: dict | None, repaired: dict | None) -> dict:
    """Never drop a custom kernel or attention rewrite just to pass rel_loss.

    Repair may change compile flags, mark_dynamic, autograd notes — not the
    insert contract. If the LLM omitted replace_pattern / apply_rewrite, put
    the failed landing actions back at the front.
    """
    failed = failed or {}
    out = strip_invented_action_keys(repaired or {})
    keep = _landing_ops(failed)
    if not keep:
        return out
    catalog_rw = {
        str(a.get("rewrite") or "")
        for a in keep
        if a.get("op") == "apply_rewrite" and str(a.get("rewrite") or "") in _CATALOG_HOOKS
    }
    acts = [dict(a) for a in (out.get("actions") or [])]
    if catalog_rw:
        acts = [
            a
            for a in acts
            if not (
                a.get("op") == "apply_rewrite"
                and str(a.get("rewrite") or "") not in catalog_rw
                and str(a.get("rewrite") or "")
                not in ("fused_sdpa_attn", "sdpa_rewrite", "__generate__", "generate")
            )
        ]
    have = {_action_sig(a) for a in acts if str(a.get("op") or "") in _LANDING_OPS}
    missing = [a for a in keep if _action_sig(a) not in have]
    if missing:
        kept_ops = {str(a.get("op") or "") for a in acts if str(a.get("op") or "") in _LANDING_OPS}
        if not kept_ops:
            missing = keep
        acts = missing + [a for a in acts if str(a.get("op") or "") not in _LANDING_OPS]
        note = str(out.get("note") or "")
        out["note"] = (note + "; preserve kernel/rewrite landing").strip("; ")
    out["actions"] = acts
    if failed.get("compile_step"):
        out["compile_step"] = True
        out["compile_optimizer"] = False
        out["actions"] = [
            a for a in (out.get("actions") or []) if str(a.get("op") or "") != "compile"
        ]
    return strip_invented_action_keys(out)


class RepairAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    def repair(
        self,
        obs: Observation,
        failed_recipe: dict,
        error: str,
        history: list | None = None,
        layer: str | None = None,
    ) -> tuple[dict, str]:
        layer = (layer or failed_intervention_layer(failed_recipe, error)).upper()
        if self.llm is None:
            err = error or ""
            if layer == "A" and (
                "setattr" in err or "fullgraph" in err.lower() or "Unsupported" in err
            ):
                return _drop_fullgraph(failed_recipe), "heuristic repair: drop fullgraph"
            return (
                _keep_attempt(failed_recipe, f"heuristic: keep layer {layer} for LLM landing fix"),
                "heuristic repair: keep failed strategy",
            )

        obs_d = {
            "model": obs.model,
            "suite": obs.suite,
            "opt": obs.opt,
            "eager_ms": obs.eager_ms,
            "vanilla_ms": obs.vanilla_ms,
            "vanilla_ok": obs.vanilla_ok,
            "features": {
                k: obs.features.get(k)
                for k in (
                    "fwd_graphs",
                    "fwd_breaks",
                    "break_sample",
                    "arch_class",
                    "layerdrop_n",
                    "conv_param_frac",
                    "blocks",
                    "opt",
                )
            },
            "model_tree": obs.model_tree,
        }
        user = user_repair(
            case=obs.model,
            observation=obs_d,
            failed_recipe=failed_recipe,
            error=error,
            history=history,
            layer=layer,
        )
        raw = self.llm.chat(system_repair(layer), user)
        recipe = parse_json_object(raw)
        if not recipe or not isinstance(recipe.get("actions"), list):
            fallback = _keep_attempt(failed_recipe, "LLM parse failed; keep this attempt")
            fallback.setdefault("model", obs.model)
            fallback.setdefault("backend", "inductor")
            return fallback, raw
        recipe.setdefault("model", obs.model)
        recipe.setdefault("backend", "inductor")
        recipe = preserve_kernel_landing(failed_recipe, recipe)
        return recipe, raw
