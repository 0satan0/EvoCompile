#!/usr/bin/python3
"""Print retrieval features for a case: architecture, optimizer, param shape, fwd breaks.

These are the observables scheme_index should key on — not the model name.

Usage:
  source docker/gko_env.sh
  CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 eval/inspect_features.py --case BertForMaskedLM
  CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 eval/inspect_features.py --case deit_tiny_patch16_224.fb_in1k --explain --prof
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch._dynamo.testing import reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs

GKO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GKO))
sys.path.insert(0, str(GKO / "eval"))

from run_agent_cases import _detect_suite, _load, _setup_cwd  # noqa: E402


def _opt_kind(opt) -> str:
    if opt is None:
        return "none"
    name = type(opt).__name__
    inner = getattr(opt, "optimizer", None) or getattr(opt, "_optimizer", None)
    if inner is not None and hasattr(inner, "step"):
        return f"{name}/{type(inner).__name__}"
    return name


def _layerdrop_n(model: nn.Module) -> int:
    n = 0
    for m in model.modules():
        if hasattr(m, "layerdrop"):
            n += 1
    return n


def _embed_hidden(model: nn.Module):
    cfg = getattr(model, "config", None)
    out = {}
    if cfg is not None:
        for k in (
            "hidden_size",
            "d_model",
            "n_embd",
            "embed_dim",
            "num_hidden_layers",
            "num_layers",
            "n_layer",
            "vocab_size",
            "model_type",
        ):
            if hasattr(cfg, k):
                out[k] = getattr(cfg, k)
    for k in ("embed_dim", "num_features", "num_channels"):
        if hasattr(model, k) and k not in out:
            try:
                out[k] = int(getattr(model, k))
            except Exception:
                pass
    return out


def _param_stats(model: nn.Module) -> dict:
    n_param = 0
    n_tensors = 0
    by = collections.Counter()
    bytes_by = collections.Counter()
    small = 0  # tensors with < 4096 elems (Adam foreach launch candidates)
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n_tensors += 1
        n = p.numel()
        n_param += n
        if n < 4096:
            small += 1
        # classify by parent module type of this param... skip, use ndim
        if p.ndim == 4:
            by["conv"] += 1
            bytes_by["conv"] += n
        elif p.ndim == 2:
            by["linear"] += 1
            bytes_by["linear"] += n
        elif p.ndim == 1:
            by["vec"] += 1
            bytes_by["vec"] += n
        else:
            by["other"] += 1
            bytes_by["other"] += n
    tot = n_param or 1
    return {
        "n_params_m": round(n_param / 1e6, 3),
        "n_tensors": n_tensors,
        "n_small_tensors": small,
        "small_tensor_frac": round(small / max(n_tensors, 1), 3),
        "conv_param_frac": round(bytes_by["conv"] / tot, 3),
        "linear_param_frac": round(bytes_by["linear"] / tot, 3),
        "n_conv_tensors": by["conv"],
        "n_linear_tensors": by["linear"],
    }


def _repeat_blocks(model: nn.Module) -> dict:
    counts = collections.Counter(type(m).__name__ for m in model.modules())
    keep = {}
    for name, n in counts.most_common():
        if n >= 2 and any(
            s in name
            for s in (
                "Layer",
                "Block",
                "Bottleneck",
                "BasicBlock",
                "Encoder",
                "Attention",
                "Transformer",
            )
        ):
            keep[name] = n
    return keep


def _arch_class(stats: dict, blocks: dict, hidden: dict, conv_frac: float) -> str:
    if conv_frac >= 0.35:
        return "cnn"
    names = " ".join(blocks)
    if any(s in names for s in ("VisionTransformer", "Block", "Swin", "DeiT")) or hidden.get(
        "embed_dim"
    ):
        embed = hidden.get("embed_dim") or hidden.get("hidden_size") or hidden.get("n_embd") or 0
        try:
            embed = int(embed)
        except Exception:
            embed = 0
        if embed and embed <= 256:
            return "vit_tiny"
        if "Block" in names or embed:
            return "vit_base"
    if any(s in names for s in ("BertLayer", "GPTBlock", "GPT2Block", "Llama", "DecoderLayer", "T5Block")):
        n_layer = hidden.get("num_hidden_layers") or hidden.get("n_layer") or hidden.get("num_layers")
        try:
            n_layer = int(n_layer) if n_layer is not None else max(blocks.values() or [0])
        except Exception:
            n_layer = max(blocks.values() or [0])
        mt = str(hidden.get("model_type") or "")
        if "distil" in mt or n_layer <= 6:
            return "transformer_skinny"
        if n_layer >= 20:
            return "transformer_deep"
        return "transformer_base"
    if conv_frac >= 0.15:
        return "hybrid_cnn"
    return "other"


def _input_layout(inputs) -> dict:
    """How example inputs are packed: dict vs tuple, keys, first-tensor ndim."""

    def _first_tensor(xs):
        for x in xs:
            if torch.is_tensor(x):
                return x
            if isinstance(x, dict):
                t = _first_tensor(x.values())
                if t is not None:
                    return t
            if isinstance(x, (list, tuple)):
                t = _first_tensor(x)
                if t is not None:
                    return t
        return None

    if isinstance(inputs, dict):
        t = _first_tensor(inputs.values())
        keys = list(inputs.keys())
        return {
            "input_kind": "dict",
            "input_keys": [str(k) for k in keys[:12]],
            "input_ndim": None if t is None else int(t.ndim),
            "has_input_ids": "input_ids" in inputs,
        }
    seq = inputs if isinstance(inputs, (list, tuple)) else (inputs,)
    t = _first_tensor(seq)
    return {
        "input_kind": "tuple",
        "input_keys": [],
        "input_ndim": None if t is None else int(t.ndim),
        "has_input_ids": False,
        "n_inputs": len(seq),
    }


def _fwd_explain(model, inputs, amp: bool):
    cloned = clone_inputs(inputs)

    def fwd():
        ctx = torch.cuda.amp.autocast(enabled=amp and torch.cuda.is_available())
        with ctx:
            if isinstance(cloned, dict):
                pred = model(**cloned)
            else:
                pred = model(*cloned)
            if hasattr(pred, "loss") and pred.loss is not None:
                return pred.loss
            try:
                return reduce_to_scalar_loss(pred)
            except Exception:
                return pred

    try:
        expl = torch._dynamo.explain(fwd)()
        reasons = []
        for r in (getattr(expl, "break_reasons", None) or [])[:8]:
            reasons.append(str(r)[:160])
        out = {
            "fwd_graphs": expl.graph_count,
            "fwd_breaks": expl.graph_break_count,
            "break_sample": reasons,
        }
        try:
            from agent.break_kinds import classify_breaks

            out.update(classify_breaks(reasons))
        except Exception:
            pass
        return out
    except Exception as e:
        return {"fwd_graphs": None, "fwd_breaks": None, "explain_err": f"{type(e).__name__}: {e}"}


def _prof_buckets(model, inputs, amp: bool) -> dict:
    from torch.profiler import ProfilerActivity, profile

    from filter_feedback import summarize_profiler_obj

    cloned = clone_inputs(inputs)
    cmodel = torch.compile(model, backend="inductor", fullgraph=False)

    def step():
        ctx = torch.cuda.amp.autocast(enabled=amp and torch.cuda.is_available())
        with ctx:
            if isinstance(cloned, dict):
                pred = cmodel(**cloned)
            else:
                pred = cmodel(*cloned)
            if hasattr(pred, "loss") and pred.loss is not None:
                loss = pred.loss
            else:
                try:
                    loss = reduce_to_scalar_loss(pred)
                except Exception:
                    loss = pred
            if torch.is_tensor(loss):
                loss.backward()

    for _ in range(2):
        step()
        cmodel.zero_grad(set_to_none=True)
    acts = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "CUDA"):
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, record_shapes=False) as prof:
        step()
    return summarize_profiler_obj(prof)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--suite", default="")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--prof", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    _setup_cwd()
    suite = _detect_suite(args.case, args.suite)
    amp = False if args.no_amp else suite in ("huggingface", "timm")
    bm, model, inputs, opt = _load(args.case, suite=args.suite)
    stats = _param_stats(model)
    blocks = _repeat_blocks(model)
    hidden = _embed_hidden(model)
    feats = {
        "model": args.case,
        "suite": suite,
        "amp": amp,
        "opt": _opt_kind(opt),
        "layerdrop_n": _layerdrop_n(model),
        "arch_class": _arch_class(stats, blocks, hidden, stats["conv_param_frac"]),
        "hidden": hidden,
        "blocks": blocks,
        **stats,
        **_input_layout(inputs),
    }
    if args.explain:
        feats.update(_fwd_explain(model, inputs, amp))
    if args.prof:
        try:
            feats["profiler"] = _prof_buckets(model, inputs, amp)
        except Exception as e:
            feats["profiler_err"] = f"{type(e).__name__}: {e}"
        try:
            from torch._dynamo.utils import counters

            feats["dynamo_counters"] = {
                k: dict(v) for k, v in counters.items() if v
            }
        except Exception:
            pass
    print(json.dumps(feats, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
