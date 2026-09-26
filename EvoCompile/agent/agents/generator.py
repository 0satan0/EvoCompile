"""Optimizer LLM: scheme + observation → recipe JSON."""

from __future__ import annotations

import json
from typing import Optional

from agent.llm import LLMClient, LLMUnreachable, parse_json_object
from agent.prompts import SYSTEM_OPTIMIZE, user_optimize
from agent.state import Observation, SchemeHit


class OptimizerAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    def generate(
        self,
        obs: Observation,
        scheme: SchemeHit,
        *,
        extra: str = "",
    ) -> tuple[dict, str]:
        if self.llm is None:
            from agent.kernel_spec import ensure_kernel_spec
            from agent.memory_tree import normalize_s6_recipe

            hint = dict(scheme.recipe_hint)
            hint = normalize_s6_recipe(hint, scheme.name, obs.model)
            return ensure_kernel_spec(hint, obs, scheme.name), "heuristic: retrieved recipe_hint (no LLM)"

        obs_d = {
            "model": obs.model,
            "suite": obs.suite,
            "amp": obs.amp,
            "opt": obs.opt,
            "eager_ms": obs.eager_ms,
            "vanilla_ms": obs.vanilla_ms,
            "vanilla_ok": obs.vanilla_ok,
            "vanilla_err_tail": (obs.vanilla.err or "")[-800:],
            "features": _compact_features(obs.features),
            "model_tree": (obs.model_tree or "")[:2500],
        }
        sd = scheme.to_dict()
        sd.pop("similar", None)
        cards = sd.pop("case_cards", None) or []
        extra_block = extra
        if cards:
            extra_block = (
                "SCHEME USAGE EXAMPLES (expert experience from OTHER cases — "
                "USAGE ILLUSTRATIONS only; adapt modules/mask/window/bias to THIS "
                "observation; keep THIS case name; do not copy catalog kernel ids):\n"
                + json.dumps(cards[:3], indent=2, default=str, ensure_ascii=False)[:5000]
                + "\n"
                + extra
            )
        user = user_optimize(
            case=obs.model,
            observation=obs_d,
            scheme=sd,
            extra=extra_block,
        )
        raw = ""
        try:
            raw = self.llm.chat(SYSTEM_OPTIMIZE, user)
        except LLMUnreachable:
            raise
        except Exception as e:
            hint = dict(scheme.recipe_hint)
            hint["note"] = (hint.get("note") or "") + f"; LLM call failed: {type(e).__name__}"
            from agent.kernel_spec import ensure_kernel_spec
            from agent.memory_tree import normalize_s6_recipe

            return ensure_kernel_spec(
                normalize_s6_recipe(hint, scheme.name, obs.model), obs, scheme.name
            ), f"LLM_ERROR: {e}"
        recipe = parse_json_object(raw)
        if not recipe or not isinstance(recipe.get("actions"), list):
            hint = dict(scheme.recipe_hint)
            hint["note"] = (hint.get("note") or "") + "; LLM parse failed, used hint"
            from agent.kernel_spec import ensure_kernel_spec
            from agent.memory_tree import normalize_s6_recipe

            return ensure_kernel_spec(
                normalize_s6_recipe(hint, scheme.name, obs.model), obs, scheme.name
            ), raw
        recipe.setdefault("model", obs.model)
        recipe.setdefault("backend", "inductor")
        from agent.kernel_spec import ensure_kernel_spec
        from agent.memory_tree import normalize_s6_recipe

        recipe = normalize_s6_recipe(recipe, scheme.name, obs.model)
        return ensure_kernel_spec(recipe, obs, scheme.name), raw


def _compact_features(feats: dict | None) -> dict:
    feats = dict(feats or {})
    out = {
        k: feats.get(k)
        for k in (
            "opt",
            "arch_class",
            "layerdrop_n",
            "fwd_graphs",
            "fwd_breaks",
            "conv_param_frac",
            "linear_param_frac",
            "n_tensors",
            "n_params_m",
            "blocks",
            "hidden",
            "inductor_extern",
            "inductor_triton",
            "inductor_custom",
            "aten_ops",
            "trace_hints",
            "break_kinds",
            "input_kind",
            "has_input_ids",
            "needs_mark_dynamic",
            "leftover_source",
        )
        if k in feats
    }
    sample = feats.get("break_sample") or []
    if sample:
        out["break_sample"] = [str(x)[:120] for x in sample[:8]]
    channels = feats.get("leftover_channels") or {}
    official = channels.get("official_step") if isinstance(channels, dict) else None
    if isinstance(official, dict) and (
        official.get("inductor_extern") or official.get("inductor_triton")
    ):
        out["leftover_channels"] = {
            "official_step": {
                k: official.get(k)
                for k in ("inductor_extern", "inductor_triton", "inductor_custom", "aten_ops")
                if official.get(k) is not None
            }
        }
    prof = feats.get("profiler")
    if isinstance(prof, dict):
        share = prof.get("cuda_time_share_pct") or {}
        out["profiler"] = {"cuda_time_share_pct": {k: share[k] for k in list(share)[:8]}}
    return out
