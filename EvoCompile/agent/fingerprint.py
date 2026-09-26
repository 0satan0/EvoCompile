"""Compact task fingerprint: retrieval / similarity / evolve share this vector.

Never send model source, full recipes, or the whole case library to an LLM.
Distance and hard predicates run in Python on this dict only.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.break_kinds import classify_breaks
from agent.state import Observation


def _has_unfused(feats: dict) -> bool:
    from agent.fusion_sites import has_unfused_epilogue, i_channel_for_s6

    return has_unfused_epilogue(i_channel_for_s6(feats))


def _unfused_bmm_attention(feats: dict) -> bool:
    from agent.fusion_sites import i_channel_for_s6, unfused_bmm_attention

    return unfused_bmm_attention(i_channel_for_s6(feats))

NUM_SCALES: dict[str, tuple[float, float]] = {
    # key: (typical max for L1 scale, weight)
    "eager_ms": (400.0, 0.6),
    "vanilla_ms": (400.0, 0.8),
    "ratio": (4.0, 1.4),
    "fwd_graphs": (40.0, 1.2),
    "fwd_breaks": (40.0, 1.4),
    "layerdrop_n": (4.0, 2.0),
    "conv_param_frac": (1.0, 1.8),
    "linear_param_frac": (1.0, 1.2),
    "n_tensors": (1200.0, 1.0),
    "n_params_m": (400.0, 1.0),
    "hidden": (4096.0, 1.2),
    "n_layer": (48.0, 1.2),
    "vocab_size": (250000.0, 0.8),
    "sync": (100.0, 1.4),
    "dense_n": (80.0, 1.5),
}

BIN_WEIGHTS: dict[str, float] = {
    "eager_ok": 1.5,
    "vanilla_ok": 1.5,
    "adam": 1.8,
    "is_llm": 1.6,
    "has_lstm": 1.2,
    "has_setattr": 1.0,
    "has_nonzero": 1.0,
    "has_dynamic": 1.2,
    "has_side_effect": 1.1,
    "needs_mark_dynamic": 1.0,
    "zero_break": 1.4,
    "has_unfused_epilogue": 1.3,
    "unfused_bmm_attention": 1.5,
    "leftover_step": 1.6,
    "leftover_python": 1.2,
    "has_moco": 1.0,
}

CAT_WEIGHTS: dict[str, float] = {
    "arch_class": 1.4,
    "model_type": 1.0,
    "suite": 0.4,
}


def opt_is_adam(opt: str) -> bool:
    o = opt or ""
    if any(x in o for x in ("SGD", "RMSprop", "Combined", "DPOptimizer", "Adagrad")):
        return False
    return "Adam" in o


# Repeat-block names that are real depth, not LayerNorm / Dropout counts.
_LAYER_BLOCKS = (
    "TransformerBlock",
    "BertLayer",
    "RobertaLayer",
    "AlbertLayer",
    "DebertaLayer",
    "GPT2Block",
    "GPTNeoBlock",
    "GPTJBlock",
    "LlamaDecoderLayer",
    "Qwen2DecoderLayer",
    "GemmaDecoderLayer",
    "BartEncoderLayer",
    "BartDecoderLayer",
    "T5Block",
    "T5Layer",
    "ViTLayer",
    "SwinTransformerBlock",
    "CLIPEncoderLayer",
    "FNetLayer",
    "XLNetLayer",
    "DecoderLayer",
    "EncoderLayer",
    "Bottleneck",
    "BasicBlock",
    "DenseLayer",
    "_DenseLayer",
    "InvertedResidual",
    "YOLOLayer",
)
_NOT_DEPTH = (
    "LayerNorm",
    "Dropout",
    "Linear",
    "Embedding",
    "Conv",
    "ReLU",
    "GELU",
    "SiLU",
    "Attention",
    "MultiHead",
    "Softmax",
)


def hidden_size(feats: dict) -> Optional[int]:
    """None if the model has no config hidden; 0 is not 'skinny'."""
    h = feats.get("hidden") or {}
    if not isinstance(h, dict):
        try:
            n = int(h)
            return n
        except (TypeError, ValueError):
            return None
    for k in ("hidden_size", "d_model", "n_embd", "embed_dim"):
        v = h.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return None


def n_layer(feats: dict) -> int:
    h = feats.get("hidden") or {}
    if isinstance(h, dict):
        for k in ("num_hidden_layers", "n_layer", "num_layers"):
            v = h.get(k)
            if v is not None:
                try:
                    n = int(v)
                    if n > 0:
                        return n
                except (TypeError, ValueError):
                    pass
    blocks = feats.get("blocks") or {}
    if not isinstance(blocks, dict) or not blocks:
        return 0
    for name in _LAYER_BLOCKS:
        if name in blocks:
            n = _as_int(blocks[name])
            if n > 0:
                return n
    vals = []
    for k, v in blocks.items():
        if any(s in str(k) for s in _NOT_DEPTH):
            continue
        n = _as_int(v)
        if n > 0:
            vals.append(n)
    return max(vals) if vals else 0


def is_skinny(*, model_type: str, nlay: int, ratio, hidden: Optional[int], n_tensors: int, vanilla_ms) -> bool:
    """Distil / tiny hidden / huge tensor count / deep-but-tiny-ms. Missing hidden is NOT skinny."""
    mt = (model_type or "").lower()
    if "distil" in mt:
        return True
    if nlay > 0 and nlay <= 6 and ratio is not None and ratio >= 2.3:
        return True
    if hidden is not None and 0 < hidden <= 256:
        return True
    if n_tensors > 500:
        return True
    if nlay >= 20 and (vanilla_ms or 0) < 80:
        return True
    return False


def is_llm_name(name: str) -> bool:
    return any(x in (name or "") for x in ("Llama", "Qwen", "gemma", "GPTJ", "Mistral", "gpt-oss"))


def leftover_step_from_ctx(d: dict) -> bool:
    """Retrieve compile(train_step) when leftover after compile(model) is the step.

    Naive baseline stays torch.compile(model). This flag is only the retrieval
    key for wrapping fwd+bwd+opt (official Dynamo boundary).

    True when:
    - FNet / cuFFT fallback (fullgraph FFT is wrong; wrap around the vendor op)
    - HF_LLM 0-break (compile(model)+outer AMP leaves autocast/bwd in Python)
    - 0-break fused module with profiler other% (Python / opt.step) still high

    False when: moco/contrastive wrap regression, YOLO rewrite cases, skinny
    transformers, tiny steps, S3-fc classifier-head (that leftover is Adam
    launches, not train_step glue).
    """
    name = str(d.get("name") or "").lower()
    if d.get("has_moco") or "moco" in name:
        return False
    if d.get("has_yolo") or "yolo" in name:
        return False
    if d.get("hot_breaks") or d.get("skinny"):
        return False
    vanilla_ms = d.get("vanilla_ms")
    try:
        vms = float(vanilla_ms) if vanilla_ms is not None else 0.0
    except (TypeError, ValueError):
        vms = 0.0
    if 0 < vms < 15:
        return False
    if d.get("has_fnet"):
        return True
    if d.get("is_llm") and d.get("zero_break"):
        return True
    conv = _as_float(d.get("conv"))
    lin = _as_float(d.get("lin"))
    n_layer = 0
    try:
        n_layer = int(d.get("n_layer") or 0)
    except (TypeError, ValueError):
        n_layer = 0
    # VGG/AlexNet classifier-head: leftover is Adam launches, not train_step glue.
    # Transformers also have lin≈1 and conv=0 — do not treat those as S3-fc.
    if conv < 0.3 and lin >= 0.5 and n_layer < 8:
        return False
    leftover_python = bool(d.get("leftover_python")) or _as_float(d.get("sync")) >= 50
    if d.get("zero_break") and leftover_python and vms >= 20:
        return True
    return False


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def match_context(obs: Observation) -> dict:
    """Full predicate context for the retrieval tree (not sent to the LLM)."""
    name = obs.model
    feats = obs.features or {}
    opt = obs.opt or feats.get("opt") or ""
    adam = opt_is_adam(opt)
    conv = _as_float(feats.get("conv_param_frac"))
    lin = _as_float(feats.get("linear_param_frac"))
    breaks = feats.get("fwd_breaks")
    graphs = feats.get("fwd_graphs")
    layerdrop = _as_int(feats.get("layerdrop_n"))
    arch = str(feats.get("arch_class") or "")
    blocks = feats.get("blocks") or {}
    n_tensors = _as_int(feats.get("n_tensors"))
    n_params = _as_float(feats.get("n_params_m"))
    hidden = hidden_size(feats)
    nlay = n_layer(feats)
    vocab = 0
    try:
        vocab = int((feats.get("hidden") or {}).get("vocab_size") or 0)
    except (TypeError, ValueError):
        pass
    model_type = str((feats.get("hidden") or {}).get("model_type") or "")
    ratio = obs.speedup_vanilla_over_eager()
    vanilla_ms = obs.vanilla_ms
    eager_ms = obs.eager_ms
    prof = feats.get("profiler") or {}
    share = (prof.get("cuda_time_share_pct") or {}) if isinstance(prof, dict) else {}
    sync = _as_float(share.get("other"))
    breaks_s = " ".join(str(x) for x in (feats.get("break_sample") or []))
    tagged = classify_breaks(feats.get("break_sample") or [])
    if feats.get("has_side_effect") is True:
        tagged["has_side_effect"] = True
        tagged["has_print_break"] = True
    dense_n = _as_int(blocks.get("DenseLayer") or blocks.get("_DenseLayer"))
    prof_s = str(prof)
    b = _as_int(breaks)
    zero_break = breaks in (0, None) or (graphs == 1 and b <= 1)
    vanilla_gt_eager = bool(eager_ms and vanilla_ms and vanilla_ms > eager_ms)
    vanilla_slower = bool(
        eager_ms and vanilla_ms and vanilla_ms > eager_ms * 1.02
    )
    side = bool(tagged.get("has_side_effect"))
    hot_tokens = any(x in breaks_s for x in ("setattr", "nonzero", "Dynamic")) or side
    hot_breaks = b >= 10 or (
        b >= 5
        and ratio is not None
        and ratio <= 1.08
        and hot_tokens
    ) or (
        side
        and b >= 2
        and (ratio is None or ratio <= 1.08)
    )
    skinny = is_skinny(
        model_type=model_type,
        nlay=nlay,
        ratio=ratio,
        hidden=hidden,
        n_tensors=n_tensors,
        vanilla_ms=vanilla_ms,
    )
    hint_s = " ".join(str(h) for h in (feats.get("trace_hints") or feats.get("hints") or []))
    unique_g = _as_int(
        feats.get("unique_graphs")
        or ((feats.get("dynamo_counters") or {}).get("stats") or {}).get("unique_graphs")
    )
    needs_mark_dynamic = (
        "many_static_shape_guards" in hint_s
        or unique_g >= 8
        or bool(feats.get("needs_mark_dynamic"))
    )
    input_kind = str(feats.get("input_kind") or "")
    input_keys = list(feats.get("input_keys") or [])
    has_input_ids = bool(feats.get("has_input_ids")) or "input_ids" in input_keys
    input_ndim = feats.get("input_ndim")
    has_moco = "moco" in name.lower()
    leftover_python = sync >= 50
    ctx = {
        "name": name,
        "suite": obs.suite or "",
        "opt": opt,
        "eager_ok": bool(obs.eager.ok),
        "vanilla_ok": obs.vanilla_ok,
        "eager_ms": eager_ms,
        "vanilla_ms": vanilla_ms,
        "ratio": ratio,
        "adam": adam,
        "conv": conv,
        "lin": lin,
        "fwd_breaks": breaks,
        "fwd_graphs": graphs,
        "layerdrop_n": layerdrop,
        "arch": arch,
        "arch_class": arch,
        "n_tensors": n_tensors,
        "n_params": n_params,
        "n_params_m": n_params,
        "hidden": hidden,
        "n_layer": nlay,
        "vocab": vocab,
        "vocab_size": vocab,
        "model_type": model_type,
        "sync": sync,
        "breaks_s": breaks_s,
        "dense_n": dense_n,
        "is_llm": is_llm_name(name),
        "zero_break": zero_break,
        "vanilla_slower": vanilla_slower,
        "vanilla_gt_eager": vanilla_gt_eager,
        "hot_breaks": hot_breaks,
        "skinny": skinny,
        "has_lstm": bool(tagged.get("has_lstm")) or "LSTM" in breaks_s,
        "has_rnn": bool(tagged.get("has_rnn")) or "RNN" in breaks_s,
        "has_cudnn": "cuDNN" in prof_s,
        "has_setattr": bool(tagged.get("has_setattr")) or "setattr" in breaks_s or "problem_type" in breaks_s,
        "has_nonzero": bool(tagged.get("has_nonzero")) or "nonzero" in breaks_s,
        "has_dynamic": bool(tagged.get("has_dynamic")) or "Dynamic" in breaks_s,
        "has_item": bool(tagged.get("has_item")) or "item" in breaks_s,
        "has_pad": "pad" in breaks_s.lower(),
        "has_yolo": "YOLO" in breaks_s or "yolo" in name.lower() or "create_grids" in breaks_s,
        "has_speech": "speech" in name.lower() or "pad" in breaks_s.lower(),
        "has_embedbag": bool(tagged.get("has_embedbag")) or "EmbeddingBag" in breaks_s,
        "has_combined": "Combined" in opt,
        "has_fnet": "FNet" in name or "GoogleFnet" in name or "fnet" in model_type.lower(),
        "tiny_vit": arch == "vit_tiny"
        or (hidden is not None and hidden <= 192 and "cnn" not in arch),
        "has_unfused_epilogue": _has_unfused(feats),
        "unfused_bmm_attention": _unfused_bmm_attention(feats),
        "has_side_effect": side,
        "has_print_break": side,
        "has_vendor_break": bool(tagged.get("has_vendor_break")),
        "break_kinds": list(tagged.get("break_kinds") or feats.get("break_kinds") or []),
        "needs_mark_dynamic": needs_mark_dynamic,
        "input_kind": input_kind,
        "input_keys": input_keys,
        "has_input_ids": has_input_ids,
        "input_ndim": input_ndim,
        "unique_graphs": unique_g,
        "has_moco": has_moco,
        "leftover_python": leftover_python,
        "leftover_source": str(feats.get("leftover_source") or "naive"),
    }
    ctx["leftover_step"] = leftover_step_from_ctx(ctx)
    return ctx


def compact_fp(obs: Observation) -> dict:
    """~20 scalars/flags. This is what similarity/evolve send to the LLM."""
    ctx = match_context(obs)
    return compact_fp_from_ctx(ctx, case=obs.model, suite=obs.suite or "")


def compact_fp_from_ctx(ctx: dict, *, case: str = "", suite: str = "") -> dict:
    def _num(k: str):
        v = ctx.get(k)
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        try:
            f = float(v)
            if f == int(f) and k not in (
                "conv_param_frac",
                "linear_param_frac",
                "ratio",
                "n_params_m",
                "conv",
                "lin",
            ):
                return int(f)
            return round(f, 4)
        except (TypeError, ValueError):
            return v

    return {
        "case": case or ctx.get("name") or "",
        "suite": suite or ctx.get("suite") or "",
        "opt": ctx.get("opt") or "",
        "arch_class": ctx.get("arch_class") or "",
        "model_type": ctx.get("model_type") or "",
        "eager_ok": ctx.get("eager_ok"),
        "vanilla_ok": ctx.get("vanilla_ok"),
        "eager_ms": _num("eager_ms"),
        "vanilla_ms": _num("vanilla_ms"),
        "ratio": None if ctx.get("ratio") is None else round(float(ctx["ratio"]), 3),
        "fwd_graphs": ctx.get("fwd_graphs"),
        "fwd_breaks": ctx.get("fwd_breaks"),
        "layerdrop_n": ctx.get("layerdrop_n"),
        "conv_param_frac": _num("conv"),
        "linear_param_frac": _num("lin"),
        "n_tensors": ctx.get("n_tensors"),
        "n_params_m": _num("n_params"),
        "hidden": ctx.get("hidden"),
        "n_layer": ctx.get("n_layer"),
        "vocab_size": ctx.get("vocab"),
        "sync": _num("sync"),
        "dense_n": ctx.get("dense_n"),
        "adam": bool(ctx.get("adam")),
        "is_llm": bool(ctx.get("is_llm")),
        "zero_break": bool(ctx.get("zero_break")),
        "has_lstm": bool(ctx.get("has_lstm")),
        "has_setattr": bool(ctx.get("has_setattr")),
        "has_nonzero": bool(ctx.get("has_nonzero")),
        "has_dynamic": bool(ctx.get("has_dynamic")),
        "has_side_effect": bool(ctx.get("has_side_effect")),
        "needs_mark_dynamic": bool(ctx.get("needs_mark_dynamic")),
        "has_unfused_epilogue": bool(ctx.get("has_unfused_epilogue")),
        "unfused_bmm_attention": bool(ctx.get("unfused_bmm_attention")),
        "leftover_step": bool(ctx.get("leftover_step")),
        "leftover_source": str(ctx.get("leftover_source") or "naive"),
        "leftover_python": bool(ctx.get("leftover_python")),
        "has_moco": bool(ctx.get("has_moco")),
        "has_fnet": bool(ctx.get("has_fnet")),
        "input_kind": ctx.get("input_kind") or "",
        "has_input_ids": bool(ctx.get("has_input_ids")),
    }


def context_from_fp(fp: dict) -> dict:
    """Rebuild a predicate context from a stored compact card (missing keys default)."""
    name = str(fp.get("case") or fp.get("name") or "")
    opt = str(fp.get("opt") or "")
    conv = _as_float(fp.get("conv_param_frac"))
    lin = _as_float(fp.get("linear_param_frac"))
    breaks = fp.get("fwd_breaks")
    graphs = fp.get("fwd_graphs")
    layerdrop = _as_int(fp.get("layerdrop_n"))
    arch = str(fp.get("arch_class") or "")
    n_tensors = _as_int(fp.get("n_tensors"))
    n_params = _as_float(fp.get("n_params_m"))
    hidden = fp.get("hidden")
    if hidden is not None:
        hidden = _as_int(hidden)
        if hidden <= 0:
            hidden = None
    nlay = _as_int(fp.get("n_layer"))
    vocab = _as_int(fp.get("vocab_size"))
    model_type = str(fp.get("model_type") or "")
    ratio = fp.get("ratio")
    if ratio is not None:
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            ratio = None
    vanilla_ms = fp.get("vanilla_ms")
    eager_ms = fp.get("eager_ms")
    if vanilla_ms is not None:
        vanilla_ms = _as_float(vanilla_ms)
    if eager_ms is not None:
        eager_ms = _as_float(eager_ms)
    sync = _as_float(fp.get("sync"))
    dense_n = _as_int(fp.get("dense_n"))
    adam = fp.get("adam")
    if adam is None:
        adam = opt_is_adam(opt)
    b = _as_int(breaks)
    zero_break = fp.get("zero_break")
    if zero_break is None:
        zero_break = breaks in (0, None) or (graphs == 1 and b <= 1)
    vanilla_gt_eager = bool(eager_ms and vanilla_ms and vanilla_ms > eager_ms)
    vanilla_slower = bool(eager_ms and vanilla_ms and vanilla_ms > eager_ms * 1.02)
    skinny = is_skinny(
        model_type=model_type,
        nlay=nlay,
        ratio=ratio,
        hidden=hidden,
        n_tensors=n_tensors,
        vanilla_ms=vanilla_ms,
    )
    is_llm = fp.get("is_llm")
    if is_llm is None:
        is_llm = is_llm_name(name)
    leftover_python = bool(fp.get("leftover_python")) or sync >= 50
    ctx = {
        "name": name,
        "suite": str(fp.get("suite") or ""),
        "opt": opt,
        "eager_ok": bool(fp.get("eager_ok", True)),
        "vanilla_ok": fp.get("vanilla_ok", True),
        "eager_ms": eager_ms,
        "vanilla_ms": vanilla_ms,
        "ratio": ratio,
        "adam": bool(adam),
        "conv": conv,
        "lin": lin,
        "fwd_breaks": breaks,
        "fwd_graphs": graphs,
        "layerdrop_n": layerdrop,
        "arch": arch,
        "arch_class": arch,
        "n_tensors": n_tensors,
        "n_params": n_params,
        "n_params_m": n_params,
        "hidden": hidden,
        "n_layer": nlay,
        "vocab": vocab,
        "vocab_size": vocab,
        "model_type": model_type,
        "sync": sync,
        "breaks_s": str(fp.get("breaks_s") or ""),
        "dense_n": dense_n,
        "is_llm": bool(is_llm),
        "zero_break": bool(zero_break),
        "vanilla_slower": vanilla_slower,
        "vanilla_gt_eager": vanilla_gt_eager,
        "hot_breaks": bool(fp.get("hot_breaks", False)),
        "skinny": skinny,
        "has_lstm": bool(fp.get("has_lstm")),
        "has_rnn": bool(fp.get("has_rnn")),
        "has_cudnn": bool(fp.get("has_cudnn")),
        "has_setattr": bool(fp.get("has_setattr")),
        "has_nonzero": bool(fp.get("has_nonzero")),
        "has_dynamic": bool(fp.get("has_dynamic")),
        "has_item": bool(fp.get("has_item")),
        "has_pad": bool(fp.get("has_pad")),
        "has_yolo": bool(fp.get("has_yolo")) or "yolo" in name.lower(),
        "has_speech": bool(fp.get("has_speech")) or "speech" in name.lower(),
        "has_embedbag": bool(fp.get("has_embedbag")),
        "has_combined": bool(fp.get("has_combined")) or "Combined" in opt,
        "has_fnet": bool(fp.get("has_fnet"))
        or "FNet" in name
        or "GoogleFnet" in name
        or "fnet" in model_type.lower(),
        "tiny_vit": arch == "vit_tiny"
        or (hidden is not None and hidden <= 192 and "cnn" not in arch),
        "has_unfused_epilogue": bool(fp.get("has_unfused_epilogue")),
        "unfused_bmm_attention": bool(fp.get("unfused_bmm_attention")),
        "has_side_effect": bool(fp.get("has_side_effect")),
        "has_print_break": bool(fp.get("has_side_effect") or fp.get("has_print_break")),
        "has_vendor_break": bool(fp.get("has_vendor_break")),
        "needs_mark_dynamic": bool(fp.get("needs_mark_dynamic")),
        "input_kind": str(fp.get("input_kind") or ""),
        "input_keys": list(fp.get("input_keys") or []),
        "has_input_ids": bool(fp.get("has_input_ids")),
        "input_ndim": fp.get("input_ndim"),
        "break_kinds": list(fp.get("break_kinds") or []),
        "unique_graphs": _as_int(fp.get("unique_graphs")),
        "has_moco": bool(fp.get("has_moco")) or "moco" in name.lower(),
        "leftover_python": leftover_python,
        "leftover_source": str(fp.get("leftover_source") or "naive"),
    }
    ctx["leftover_step"] = (
        bool(fp.get("leftover_step"))
        if fp.get("leftover_step") is not None
        else leftover_step_from_ctx(ctx)
    )
    return ctx


def fingerprint_distance(a: dict, b: dict) -> float:
    """Weighted L1 + Hamming on overlapping keys. 0 = identical, ~1 = far."""
    score = 0.0
    weight = 0.0
    for k, (scale, w) in NUM_SCALES.items():
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            continue
        try:
            score += w * min(1.0, abs(float(va) - float(vb)) / scale)
            weight += w
        except (TypeError, ValueError):
            continue
    for k, w in BIN_WEIGHTS.items():
        if k not in a or k not in b or a[k] is None or b[k] is None:
            continue
        score += w * (0.0 if bool(a[k]) == bool(b[k]) else 1.0)
        weight += w
    for k, w in CAT_WEIGHTS.items():
        sa, sb = a.get(k), b.get(k)
        if not sa or not sb:
            continue
        score += w * (0.0 if str(sa).lower() == str(sb).lower() else 1.0)
        weight += w
    if weight <= 0:
        return 1.0
    return score / weight


def case_card(entry: dict, *, dist: Optional[float] = None) -> dict:
    """One historical case for an LLM prompt: no source, no full recipe."""
    fp = entry.get("fp") or {}
    card = {
        "case": entry.get("case") or fp.get("case"),
        "scheme": entry.get("scheme"),
        "verdict": entry.get("verdict"),
        "beat_naive": entry.get("beat_naive"),
        "vs_naive_pct": entry.get("vs_naive_pct"),
        "fp": {k: v for k, v in fp.items() if k != "case" and v is not None},
    }
    if dist is not None:
        card["dist"] = round(float(dist), 3)
    return card
