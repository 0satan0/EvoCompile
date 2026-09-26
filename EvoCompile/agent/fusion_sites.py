"""I-channel fusion holes that custom Triton can actually beat official compile.

S6 / KernelWriter exists to *outrun* torch.compile (official Dynamo included):
cut HBM round-trips, fuse a split epilogue, land a vendor-class GEMM+act.

  YES  extern GEMM + *separate* Linear-act (gelu/relu/…) AND leftover triton or
       copy/HBM is material (or no profiler: classic addmm+gelu catalog hole).
       Implement with CUTLASS GEMM+epilogue (catalog addmm_gelu), not naive tl.dot.
  YES  silu and mul split across triton names (SwiGLU / SE). Keep vendor GEMM/conv.
  YES  many leftover triton_poi_* AND profiler shows launch/HBM leftover
       (triton% / copy_alloc%), not a GEMM-dominated step.
  NO   bare extern GEMM/conv with no adjacent pointwise — correct cuBLAS/cuDNN.
  NO   leftover Flash/SDPA alone — S5, not S6.
  NO   unfused bmm+softmax attention — S6-sdpa (hook + vendor FMHA / tiled FA), not S6-fuse.
  NO   already-fused silu_mul + leftover mm — vendor GEMM is the right lowering.
  NO   GEMM/conv ≥60% CUDA and leftover triton/copy tiny — naive tl.dot loses to cuBLAS.
  NO   leftover cuDNN conv — custom CUTLASS conv loses the same way.
  NO   a lone missed_gemm_epilogue hint with no I-channel names.

A site is not a scheme: Retriever still picks S6-fuse; the writer implements
the hole (catalog kernel or generated module with register()+FX pass).
Empty sites ⇒ do not retrieve or explore S6.

I-channel leftover must be the compile wrap being continued: official
``compile(train_step)`` at first S6 retrieve, then the last recipe's residual.
Do not score S6 holes off naive ``compile(model)`` leftover when an official
or overlay channel exists.
"""

from __future__ import annotations

from typing import Any, Optional

_GEMM = ("mm", "addmm", "bmm", "addmm_")
_ATTN_EXTERN = ("flash", "sdpa", "efficient_attention")
# Linear+act epilogues. silu is SwiGLU (paired with mul), not this list.
# dropout lives in embed/attn kernels; counting it as GEMM+act steals Longformer
# into addmm_gelu when softmax was stripped from leftover names.
_LINEAR_ACT = ("gelu", "relu", "sigmoid", "tanh", "softplus", "elu", "mish")
_CHAIN = _LINEAR_ACT + ("mul", "add", "softmax", "norm", "layer_norm", "rms", "silu", "dropout")
_BMM_ATTN_MIN = 8
_NON_ATTN_BMM = ("dlrm", "moco", "embedbag")


def _pairs(items) -> list[tuple[str, int]]:
    out = []
    if not items:
        return out
    for it in items:
        if isinstance(it, (list, tuple)) and len(it) >= 1:
            out.append((str(it[0]), int(it[1]) if len(it) > 1 else 1))
        else:
            out.append((str(it), 1))
    return out


def _name_tokens(n: str) -> set:
    return set(n.lower().replace("-", "_").split("_"))


def _triton_has(triton: list[tuple[str, int]], tok: str) -> bool:
    return any(tok in _name_tokens(n) for n, _ in triton)


def _poi_distinct(triton: list[tuple[str, int]]) -> list[tuple[str, int]]:
    out = []
    for n, c in triton:
        ln = n.lower()
        if "poi" in ln or "pointwise" in ln or "fused" in ln:
            out.append((n, c))
    return out


def _split_across_kernels(triton: list[tuple[str, int]], tokens: tuple[str, ...]) -> bool:
    """Tokens appear, but not already fused into a single kernel name."""
    token_sets = [_name_tokens(n) for n, _ in triton]
    if not token_sets:
        return False
    if any(all(t in ts for t in tokens) for ts in token_sets):
        return False
    return all(any(t in ts for ts in token_sets) for t in tokens)


def leftover_channel_blobs(feats: dict | None) -> list[dict]:
    """Top-level leftover plus naive/official_step dumps (token presence only)."""
    feats = feats or {}
    out: list[dict] = [feats]
    ch = feats.get("leftover_channels")
    if isinstance(ch, dict):
        for v in ch.values():
            if isinstance(v, dict):
                out.append(v)
    return out


def _joined_op_names(items) -> str:
    return " ".join(
        str(it[0] if isinstance(it, (list, tuple)) else it).lower()
        for it in (items or [])
    )


def _aten_join(feats: dict | None) -> str:
    return _joined_op_names((feats or {}).get("aten_ops"))


def _silu_mul_split(feats: dict, triton: list[tuple[str, int]]) -> bool:
    if _split_across_kernels(triton, ("silu", "mul")):
        return True
    if any("silu" in _name_tokens(n) and "mul" in _name_tokens(n) for n, _ in triton):
        return False
    aten = _aten_join(feats)
    # compile_step leftover often names silu as triton_poi_fused_N; aten still lists silu.
    return "silu" in aten and ("mul" in aten or _triton_has(triton, "mul"))


def _share(feats: dict, key: str) -> float:
    prof = feats.get("profiler") or {}
    if not isinstance(prof, dict):
        return 0.0
    share = prof.get("cuda_time_share_pct") or {}
    if not isinstance(share, dict):
        return 0.0
    try:
        return float(share.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _has_profiler(feats: dict) -> bool:
    return any(_share(feats, k) > 0 for k in ("gemm_conv", "triton", "copy_alloc", "other", "attention"))


def hardware_bound(feats: dict | None) -> bool:
    """Vendor GEMM/conv already owns the step; leftover epi too small to beat official."""
    feats = feats or {}
    if not _has_profiler(feats):
        return False
    g = _share(feats, "gemm_conv")
    t = _share(feats, "triton")
    c = _share(feats, "copy_alloc")
    return g >= 60.0 and t < 10.0 and c < 8.0


def leftover_kernel_worth(feats: dict | None) -> Optional[bool]:
    """True = leftover triton/HBM is big enough that a fused kernel can win.

    None = no profiler (caller decides per-site default).
    """
    feats = feats or {}
    if not _has_profiler(feats):
        return None
    t = _share(feats, "triton")
    c = _share(feats, "copy_alloc")
    return t >= 10.0 or c >= 8.0 or (t + c) >= 15.0


def _allow_gemm_epilogue(feats: dict) -> bool:
    if hardware_bound(feats):
        return False
    worth = leftover_kernel_worth(feats)
    if worth is False:
        return False
    return True


def _allow_launch_bound(feats: dict) -> bool:
    """Need profiler evidence. Many poi names alone is normal Inductor lowering."""
    return leftover_kernel_worth(feats) is True and not hardware_bound(feats)


_GEMM_STRATEGY = (
    "Do NOT replace cuBLAS with naive Triton tl.dot (Bert catalog 0.887x). "
    "Use catalog addmm_gelu: CUTLASS Gemm + LinearCombinationGELU on fp16 "
    "(kernels.cutlass_gemm_attn.addmm_epilogue). Autograd must save the pre-GELU "
    "z for gelu_backward — recomputing addmm in bwd loses e2e train (Bert 0.80x). "
    "register() must custom_op gko::* + register_autograd(backward, setup_context=) "
    "+ kernels.pass_install FX rewrite of every Linear+GELU site. "
    "Then compile_step=true (compile(train_step)); do not nest compile(model)."
)
_POI_STRATEGY = (
    "Fuse the leftover pointwise chain into ONE @triton.jit kernel. Keep vendor "
    "GEMM/conv/Flash as extern. Do not replace cuBLAS/cuDNN with naive tl.dot. "
    "silu*mul: catalog silu_mul. register() must custom_op gko::* + register_autograd "
    "+ pass_install FX rewrite of every matching site (loop, no break after the first). "
    "Then compile_step=true; do not nest compile(model)."
)


def sites_from_features(feats: dict | None) -> list[dict[str, Any]]:
    """Return 0+ fusion holes. Empty ⇒ no S6."""
    feats = feats or {}
    hints = [str(h) for h in (feats.get("trace_hints") or [])]
    extern = _pairs(feats.get("inductor_extern"))
    triton = _pairs(feats.get("inductor_triton"))
    en = " ".join(k.lower() for k, _ in extern)
    gemm = any(g in en for g in _GEMM)
    poi = _poi_distinct(triton)
    linear_act = [
        e
        for e in _LINEAR_ACT
        if _triton_has(triton, e)
        and not any(
            e in _name_tokens(n) and any(g in _name_tokens(n) for g in _GEMM)
            for n, _ in triton
        )
    ]
    fused_swiglu = any("silu" in _name_tokens(n) and "mul" in _name_tokens(n) for n, _ in triton)

    sites: list[dict[str, Any]] = []

    def add(*, kind: str, title: str, compute: str, catalog: str = "", extra: dict | None = None):
        rec = {
            "id": f"{kind}_{len(sites)}",
            "kind": kind,
            "title": title,
            "compute": compute,
            "catalog_fallback": catalog,
            "extern": extern[:8],
            "triton": triton[:8],
            "hints": [
                h
                for h in hints
                if "fusion" in h.lower() or "epilogue" in h.lower() or "replace_pattern" in h
            ],
        }
        if extra:
            rec.update(extra)
        sites.append(rec)

    if gemm and linear_act and _allow_gemm_epilogue(feats):
        add(
            kind="gemm_epilogue",
            title=f"extern GEMM + separate triton {linear_act[0]} (Inductor missed fused epilogue)",
            compute=_GEMM_STRATEGY,
            catalog="addmm_gelu" if "gelu" in linear_act else "",
            extra={
                "epilogues": linear_act,
                "strategy": "cutlass_gemm_epilogue",
                "backend": "cutlass",
            },
        )
    # leftover cuDNN conv is already tensor-core. Custom CUTLASS conv loses
    # the same way naive tl.dot loses to cuBLAS — not a retrieve site.

    if _silu_mul_split(feats, triton) and not hardware_bound(feats):
        add(
            kind="split_epilogue",
            title="silu and mul in different triton kernels (SwiGLU / SE split)",
            compute="One custom op: y = silu(gate) * up. " + _POI_STRATEGY,
            catalog="silu_mul",
            extra={
                "tokens": ["silu", "mul"],
                "strategy": "fuse_pointwise_keep_gemm",
                "backend": "triton",
            },
        )
    elif (
        (
            _split_across_kernels(triton, ("gelu", "dropout"))
            or _split_across_kernels(triton, ("gelu", "add"))
        )
        and _allow_launch_bound(feats)
    ):
        toks = [t for t in _CHAIN if any(t in n.lower() for n, _ in triton)]
        add(
            kind="split_epilogue",
            title="epilogue chain split across triton kernels",
            compute="Fuse the split pointwise chain compile did not horizontally fuse. " + _POI_STRATEGY,
            extra={"tokens": toks, "strategy": "fuse_pointwise_keep_gemm"},
        )

    if len(poi) >= 4 and _allow_launch_bound(feats):
        add(
            kind="launch_bound_poi",
            title=f"{len(poi)} distinct leftover triton_poi_* with material triton/HBM share",
            compute=_POI_STRATEGY,
            extra={"n_poi": len(poi), "strategy": "fuse_pointwise_keep_gemm"},
        )

    if gemm and poi and not linear_act and not fused_swiglu and leftover_kernel_worth(feats) is True:
        add(
            kind="gemm_pointwise",
            title="extern GEMM + leftover pointwise triton (unnamed epi, leftover is material)",
            compute=_POI_STRATEGY,
            extra={"strategy": "fuse_pointwise_keep_gemm"},
        )

    if any(a in en for a in _ATTN_EXTERN) and not linear_act and not any(
        s["kind"] in ("split_epilogue", "launch_bound_poi", "gemm_epilogue", "conv_epilogue")
        for s in sites
    ):
        return []

    seen = set()
    uniq = []
    for s in sites:
        if s["kind"] in seen:
            continue
        seen.add(s["kind"])
        uniq.append(s)
    return uniq


def unfused_bmm_attention(feats: dict | None) -> bool:
    """True when attention compiled to bmm+softmax, not in-tree Flash/mem-efficient.

    DistilBert AOT: extern bmm + triton softmax, no _scaled_dot_product_flash_*.
    compile(train_step) leftover often names softmax ``triton_poi_fused_N`` —
    stacked extern bmm (no flash) is still unfused MHA. dlrm/moco have a
    handful of interaction bmms without softmax — not this flag.
    """
    feats = feats or {}
    name = str(feats.get("model") or "").lower()
    if any(s in name for s in _NON_ATTN_BMM) or feats.get("has_embedbag"):
        return False
    blobs = leftover_channel_blobs(feats)
    aten = " ".join(_aten_join(b) for b in blobs)
    if any(s in aten for s in ("flash_attention", "efficient_attention")):
        return False
    extern_pairs: list[tuple[str, int]] = []
    triton_parts: list[str] = []
    for b in blobs:
        extern_pairs.extend(_pairs(b.get("inductor_extern")))
        triton_parts.append(_joined_op_names(b.get("inductor_triton")))
    n_bmm = sum(c for n, c in extern_pairs if n.lower() == "bmm")
    if n_bmm <= 0:
        return False
    triton = " ".join(triton_parts)
    if "softmax" in triton or "softmax" in aten:
        return True
    return n_bmm >= _BMM_ATTN_MIN


def has_unfused_epilogue(feats: dict | None) -> bool:
    """True iff I-channel shows a custom-Triton hole that can beat official compile."""
    return bool(sites_from_features(feats))


I_CHANNEL_KEYS = (
    "inductor_extern",
    "inductor_triton",
    "inductor_custom",
    "aten_ops",
    "trace_hints",
)


def snapshot_i_channel(feats: dict | None) -> dict:
    feats = feats or {}
    return {k: feats.get(k) for k in I_CHANNEL_KEYS}


def merge_i_channel(feats: dict | None, channel: dict | None) -> dict:
    out = dict(feats or {})
    if not isinstance(channel, dict):
        return out
    for k in I_CHANNEL_KEYS:
        if channel.get(k) is not None:
            out[k] = channel[k]
    return out


def i_channel_for_s6(feats: dict | None) -> dict:
    """I-channel of the compile S6 overlays: current wrap, else official_step, else naive.

    Collector stores compile(model) leftover at top-level (leftover_source=naive)
    and compile(train_step) leftover under leftover_channels.official_step.
    After an eval, overlay residual sets leftover_source=overlay — that wrap
    is what the next retrieve/repair continues from.
    """
    feats = feats or {}
    src = str(feats.get("leftover_source") or "naive")
    if src in ("overlay", "residual", "official_step"):
        return feats
    channels = feats.get("leftover_channels") or {}
    official = channels.get("official_step") if isinstance(channels, dict) else None
    if isinstance(official, dict) and (
        official.get("inductor_extern") or official.get("inductor_triton")
    ):
        out = merge_i_channel(feats, official)
        # Naive compile(model) profiler is a different wrap; do not let it
        # veto official leftover sites (silu_mul / addmm+gelu / bmm+softmax).
        out.pop("profiler", None)
        return out
    return feats
