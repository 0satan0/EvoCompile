"""Case-specific kernel brief: Optimizer analyzes, KernelWriter implements.

Retrieved schemes (S6-fuse / S6-sdpa) are generic. The leftover I-channel,
module tree, and attention class decide WHAT to fuse and WHERE to insert.
Catalog kernels (addmm_gelu / silu_mul / fused_sdpa) are format examples and
optional fallbacks — not the action-space enum.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.state import Observation


def _attn_modules(tree: str, model: str) -> list[str]:
    blob = f"{tree} {model}".lower()
    found: list[str] = []
    for name in (
        "LongformerSelfAttention",
        "T5Attention",
        "MT5Attention",
        "MultiHeadSelfAttention",
        "BertSelfAttention",
        "OPTAttention",
        "Attention",
    ):
        if name.lower() in blob or name in (tree or ""):
            found.append(name)
    if "longformer" in blob and "LongformerSelfAttention" not in found:
        found.append("LongformerSelfAttention")
    if ("t5" in blob or "mt5" in blob) and "T5Attention" not in found:
        found.append("T5Attention")
    if "bert_pytorch" in blob and "Attention" not in found:
        found.append("Attention")
    return found or ["*SelfAttention"]


def _case_notes(model: str, tree: str) -> str:
    n = (model or "").lower()
    blob = f"{tree} {model}".lower()
    if "longformer" in n or "longformer" in blob:
        return (
            "Sliding-window local attention, optional global tokens. "
            "Do NOT replace with dense SxS SDPA. Honor one_sided_attn_window_size; "
            "keep the global-attn branch on the original forward. "
            "The custom op MUST be torch.ops.gko.* so Inductor records inductor_custom."
        )
    if "t5" in n or "mt5" in blob:
        return (
            "T5Attention: q/k/v/o Linear, scale baked into q weights (pass scale=1.0), "
            "4D relative-position bias added to logits before softmax. "
            "Decoder causal is already in the bias — do not also set is_causal. "
            "custom_op outputs MUST .clone() — returning an input or view aliases and crashes."
        )
    if "bert_pytorch" in n:
        return (
            "BERT_pytorch Attention already has projected QKV [B,H,S,D]. "
            "Fuse score/softmax/PV only; do not re-project. "
            "Hook turns mask into additive bias via keep-mask; your op receives bias "
            "already shaped for scores, not the raw BERT mask."
        )
    if "distil" in n:
        return "DistilBert: prefer in-tree F.sdpa (vendor fused bwd), not a custom Triton FA."
    return (
        "Dense MHA leftover is extern bmm + softmax. Fuse QK^T/softmax/PV. "
        "Match eager scale (usually 1/sqrt(d_head)) and mask semantics."
    )


def _match_eager_sdpa(model: str, tree: str, modules: list[str]) -> dict[str, str]:
    """What KernelWriter must equal numerically — filled by Optimizer/synth, not guessed."""
    n = (model or "").lower()
    blob = f"{tree} {model}".lower()
    mods = ", ".join(modules) or "attention modules"
    if "longformer" in n or "longformer" in blob:
        return {
            "match_eager": (
                "Eager LongformerSelfAttention local branch: for each query, QK^T only "
                "inside one_sided_attn_window_size, apply padding/invalid mask, fp32 "
                "softmax, dropout, then P@V. Global-attn tokens stay on the original "
                "forward — do not fuse them."
            ),
            "hook_call": (
                "fused_sdpa_attn binds LongformerSelfAttention and calls "
                "torch.ops.gko.<op>(q,k,v,bias,scale,causal,window) with window from "
                "one_sided_attn_window_size. Return context only; clone if needed."
            ),
            "fx_match": (
                "If the hook is missed: rewrite leftover aten bmm / masked scores / "
                "softmax / dropout / bmm on the local window path only."
            ),
        }
    if "t5" in n or "mt5" in blob:
        return {
            "match_eager": (
                f"Eager {mods}.forward after Q/K/V projections: scores = Q @ K^T "
                "(scale already in q weights → scale=1.0), add broadcast 4D "
                "position_bias (+ mask), softmax in fp32 then cast back, attention "
                "dropout, then P @ V. Encoder self / decoder causal / cross-attn "
                "all share this pattern with different lengths."
            ),
            "hook_call": (
                "fused_sdpa_attn._call(q,k,v,bias,scale,causal,window) → your op. "
                "Return the context tensor. NEVER return an input alias — "
                "always .clone() the output of a custom_op."
            ),
            "fx_match": (
                "aten.bmm(Q,K^T) → add(position_bias) → softmax → dropout → bmm(P,V). "
                "Match all 18 T5Attention sites (loop, no break-after-first)."
            ),
        }
    if "bert_pytorch" in n:
        return {
            "match_eager": (
                "Eager BERT_pytorch Attention on already-projected Q,K,V [B,H,S,D]: "
                "scores = (Q @ K^T) * (1/sqrt(d)), apply additive keep-mask bias "
                "(masked positions ≈ -1e9 / finfo.min), softmax, dropout, P @ V. "
                "Return context only (hook ignores attn probs)."
            ),
            "hook_call": (
                "fused_sdpa_attn._bert_pytorch_attn builds bias via _bias_keep_mask(mask), "
                "then _call(q,k,v,bias,scale,0). Your op must accept that bias layout "
                "(broadcastable to scores), not re-interpret the raw Boolean mask."
            ),
            "fx_match": (
                "bmm → mul(scale) → masked_fill/add(bias) → softmax → dropout → bmm."
            ),
        }
    return {
        "match_eager": (
            f"Eager unfused attention on {mods}: scale*Q@K^T (+bias/mask), softmax, P@V. "
            "Match eager scale, mask polarity, and AMP dtype."
        ),
        "hook_call": (
            "fused_sdpa_attn._call(q,k,v,bias,scale,causal,window) → torch.ops.gko.<op>. "
            "Return context; clone outputs (custom_op forbids aliasing inputs)."
        ),
        "fx_match": "aten.bmm + softmax (+ bias/mask) + bmm on every matched site.",
    }


def synthesize_kernel_spec(obs: Observation, scheme_name: str = "") -> dict[str, Any]:
    """Deterministic brief when the Optimizer omits kernel_spec."""
    from agent.fusion_sites import i_channel_for_s6, sites_from_features

    feats = i_channel_for_s6(obs.features or {})
    sites = sites_from_features(feats)
    tree = obs.model_tree or ""
    name = scheme_name or ""
    site0 = sites[0] if sites else {}
    leftover = {
        "inductor_extern": feats.get("inductor_extern"),
        "inductor_triton": (feats.get("inductor_triton") or [])[:8],
        "aten_ops": (feats.get("aten_ops") or [])[:12],
    }
    if name == "S6-sdpa":
        modules = _attn_modules(tree, obs.model)
        match = _match_eager_sdpa(obs.model, tree, modules)
        return {
            "scheme": "S6-sdpa",
            "compute": (
                "Fused attention: scores = scale * Q @ K^T (+ bias/window mask), "
                "softmax, context = P @ V. Replaces unfused leftover bmm+softmax."
            ),
            "replace": (
                "Python attention forward on the listed modules (unfused bmm path). "
                "Not Linear+GELU. Not a generic S1 graph-break rewrite."
            ),
            "modules": modules,
            "insert": (
                "Hook those modules' .forward so compile(train_step) traces "
                "torch.ops.gko.<op>(q,k,v,...). Also register() an FX pass for "
                "aten.bmm+softmax if the hook is missed. Do NOT lower to F.sdpa "
                "unless the scheme is DistilBert/OPT in-tree SDPA — F.sdpa becomes "
                "flash/efficient ATen and inductor_custom stays empty."
            ),
            "match_eager": match["match_eager"],
            "hook_call": match["hook_call"],
            "fx_match": match["fx_match"],
            "layout": "q,k,v [B, H, S, D] (or [B,S,H,D] then transpose); fp16 AMP",
            "case_notes": _case_notes(obs.model, tree),
            "catalog_hint": None,
            "leftover": leftover,
            "format": (
                "Python 3.8 + torch 2.4: @torch.library.custom_op('gko::<name>') "
                "with Tensor annotations, register_autograd(backward, setup_context=), "
                "def register() via kernels.pass_install. "
                "FX pass: graph = gm.graph if hasattr(gm, 'graph') else gm — never gm.nodes "
                "on the Graph shim. custom_op outputs must .clone()."
            ),
        }
    compute = str(site0.get("compute") or site0.get("title") or "fuse leftover pointwise/epilogue")
    return {
        "scheme": name or "S6-fuse",
        "compute": compute,
        "replace": (
            "Leftover I-channel after official compile(train_step): "
            f"kind={site0.get('kind') or 'unknown'} tokens={site0.get('tokens') or site0.get('epilogues') or []}"
        ),
        "modules": [],
        "insert": (
            "register() FX pass rewrites EVERY matching site (loop, no break-after-first) "
            "to torch.ops.gko.<op>. Recipe is replace_pattern then compile_step; "
            "do not also compile(model)."
        ),
        "layout": "match the leftover aten/Linear dtypes and ranks",
        "case_notes": (
            f"Implement THIS leftover, not a generic catalog op, unless the hole is "
            f"exactly Linear+GELU (addmm_gelu) or silu×mul split (silu_mul)."
        ),
        "catalog_hint": site0.get("catalog_fallback") or None,
        "leftover": leftover,
        "site": {
            "kind": site0.get("kind"),
            "title": site0.get("title"),
            "strategy": site0.get("strategy"),
            "backend": site0.get("backend"),
        },
        "format": (
            "Python 3.8 + torch 2.4: custom_op gko::* + register_autograd + "
            "kernels.pass_install FX. Catalog addmm_gelu.py is the autograd calling convention."
        ),
    }


def ensure_kernel_spec(recipe: dict | None, obs: Observation, scheme_name: str = "") -> dict:
    """Keep Optimizer-written spec; fill in a synthesized one if missing."""
    out = dict(recipe or {})
    spec = out.get("kernel_spec")
    if not isinstance(spec, dict) or not spec.get("compute"):
        out["kernel_spec"] = synthesize_kernel_spec(obs, scheme_name)
    else:
        syn = synthesize_kernel_spec(obs, scheme_name)
        merged = dict(syn)
        merged.update(spec)
        if not merged.get("leftover"):
            merged["leftover"] = syn.get("leftover")
        if not merged.get("case_notes"):
            merged["case_notes"] = syn.get("case_notes")
        # Semantic contract: keep Optimizer text if present; else fill from synth.
        for key in ("match_eager", "hook_call", "fx_match"):
            if not merged.get(key) and syn.get(key):
                merged[key] = syn[key]
        out["kernel_spec"] = merged
        return out
    return out


def spec_wants_catalog(spec: Optional[dict]) -> Optional[str]:
    """Only when the brief explicitly names a catalog kernel for THIS leftover."""
    if not isinstance(spec, dict):
        return None
    hint = spec.get("catalog_hint") or spec.get("use_catalog")
    if hint in ("addmm_gelu", "silu_mul", "fused_sdpa"):
        return str(hint)
    return None
