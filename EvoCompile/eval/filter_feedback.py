#!/usr/bin/python3
"""Keep only the decision-useful slice of Dynamo / Inductor / profiler dumps.

Raw TORCH_LOGS=guards,aot_graphs,output_code and profiler tables are huge.
Almost none of the bytes are used for compile actions. This program collapses
them to the signals in memory/actionspace:

  graph break?  recompilation?  failed fusion?  custom kernel dominates?
  launch bound? already hardware-efficient?

Drop (never keep in the summary):
  - full TREE_GUARD_MANAGER / every TENSOR_MATCH stride line
  - full FX graph print (every aten node with shapes)
  - Triton kernel source, ctypes boilerplate, async_compile wrappers
  - profiler events <1% (aten::empty, record_function, ...)

Keep:
  G  unique graph-break reason + first non-torch stack frame, with counts
  D  guard *kinds* and counts; unique_graphs / calls_captured; cache_size_limit
  A  aten op histogram + dtypes seen (Half vs Float mismatch)
  I  extern_kernels.* vs triton fused *names* (not source)
  P  top CUDA kernels, grouped: gemm/conv, attention, adam, triton, copy, other

Usage:
  source docker/gko_env.sh
  /usr/bin/python3 eval/filter_feedback.py --log run/gko_eval/agent_logs/probe_aot_output.log
  CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 eval/filter_feedback.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# TORCH_LOGS lines look like:
# V0821 00:34:41.052541 14052 torch/_dynamo/guards.py:2148] [0/0] [__guards] | +- TENSOR_MATCH: ...
_LOG_TAG = re.compile(r"\[__(guards|aot_graphs|output_code|graph_breaks|recompiles|graph_code)\]\s*(.*)$")
_GUARD_KIND = re.compile(r"\+\-?\s*([A-Z][A-Z0-9_]+):")
_ATEN = re.compile(r"torch\.ops\.aten\.([A-Za-z0-9_]+)")
_DTYPE = re.compile(r"\b(f16|f32|bf16|f64|i32|i64|Half|Float|BFloat16)\b")
_EXTERN = re.compile(r"extern_kernels\.([A-Za-z0-9_]+)")
_CUSTOM_OP = re.compile(r"torch\.ops\.gko\.([A-Za-z0-9_]+)")
# Old inductor:  triton_poi_fused_x = async_compile.triton(...)
# Torch 2.8+:    async_compile.triton('triton_poi_fused_x', '''
_TRITON_NAME = re.compile(
    r"(?:(triton_(?:per_)?fused_[A-Za-z0-9_]+|triton_[A-Za-z0-9_]+)\s*=\s*async_compile\.triton"
    r"|async_compile\.triton\(\s*['\"]((?:triton_)?[A-Za-z0-9_]+)['\"])"
)
_BREAK_FROM = re.compile(r"Graph break from [`']([^`']+)[`']")
_CACHE_LIMIT = re.compile(r"cache_size_limit")
_UNIQUE_GRAPHS = re.compile(r"unique_graphs['\"]?\s*[:=]\s*(\d+)")
_CALLS_CAPTURED = re.compile(r"calls_captured['\"]?\s*[:=]\s*(\d+)")
_USER_FILE = re.compile(r"(?:file |/)([\w./-]+\.py), line (\d+)")
_SKIP_GUARD = frozenset({"RootGuardManager", "TREE_GUARD_MANAGER", "GuardManager"})

_TORCH_NOISE = (
    "torch/_dynamo",
    "torch/_inductor",
    "torch/nn/modules/module.py",
    "torch/_tensor.py",
    "torch/autograd",
    "torch/_functorch",
    "site-packages/torch/",
)


def _bucket_kernel(name: str) -> str:
    n = name.lower()
    if any(s in n for s in ("adam", "sgd", "momentum", "foreach", "optimizer")):
        return "adam"
    if any(s in n for s in ("gemm", "sgemm", "hgemm", "bmm", "mm", "addmm", "convolution", "conv", "cudnn")):
        return "gemm_conv"
    if any(s in n for s in ("sdpa", "attention", "flash", "mem_eff")):
        return "attention"
    if "triton" in n or n.startswith("triton_"):
        return "triton"
    if any(s in n for s in ("copy", "clone", "cat", "empty", "to(", "cast", "convert")):
        return "copy_alloc"
    if "inductor" in n or "compiledfunction" in n or "torch-compiled" in n:
        return "inductor_region"
    return "other"


def parse_log_text(text: str) -> Dict[str, Any]:
    guard_kinds: collections.Counter = collections.Counter()
    aten: collections.Counter = collections.Counter()
    dtypes = set()
    extern: collections.Counter = collections.Counter()
    custom: collections.Counter = collections.Counter()
    triton_kernels: collections.Counter = collections.Counter()
    breaks: collections.Counter = collections.Counter()
    user_frames: List[str] = []
    cache_limit = 0
    unique_graphs = 0
    calls_captured = 0
    tagged_lines = collections.Counter()
    raw_bytes = len(text.encode("utf-8", errors="replace"))

    for raw in text.splitlines():
        mtag = _LOG_TAG.search(raw)
        tag = mtag.group(1) if mtag else ""
        body = mtag.group(2) if mtag else raw
        if mtag:
            tagged_lines[tag] += 1

        for km in _GUARD_KIND.finditer(body):
            kind = km.group(1)
            if kind in _SKIP_GUARD:
                continue
            guard_kinds[kind] += 1

        # ATen histogram from AOT graph only — output_code reprints the same ops as Triton.
        if tag in ("aot_graphs", "graph_code", "") and "torch.ops.aten." in body:
            for am in _ATEN.finditer(body):
                aten[am.group(1)] += 1
        for dm in _DTYPE.finditer(body):
            tok = dm.group(1)
            if tok in ("Half", "f16"):
                dtypes.add("fp16")
            elif tok in ("Float", "f32"):
                dtypes.add("fp32")
            elif tok in ("BFloat16", "bf16"):
                dtypes.add("bf16")
        for em in _EXTERN.finditer(body):
            extern[em.group(1)] += 1
        for cm in _CUSTOM_OP.finditer(body):
            custom[cm.group(1)] += 1
        for tm in _TRITON_NAME.finditer(body):
            name = tm.group(1) or tm.group(2)
            if name:
                triton_kernels[name] += 1
        for bm in _BREAK_FROM.finditer(body):
            breaks[bm.group(1)] += 1
        if _CACHE_LIMIT.search(body):
            cache_limit += 1
        ug = _UNIQUE_GRAPHS.search(body)
        if ug:
            unique_graphs = max(unique_graphs, int(ug.group(1)))
        cc = _CALLS_CAPTURED.search(body)
        if cc:
            calls_captured = max(calls_captured, int(cc.group(1)))

        if "FrameSummary" in body or ".py, line" in body:
            if not any(n in body for n in _TORCH_NOISE):
                fm = _USER_FILE.search(body)
                if fm:
                    user_frames.append(f"{fm.group(1)}:{fm.group(2)}")

    prof_top, prof_buckets = parse_profiler_table(text)

    hints = []
    if "fp16" in dtypes and "fp32" in dtypes:
        hints.append("dtype_mix_fp16_fp32: AMP wrap mismatch risk (compile model + outer autocast)")
    if any("item" in k or "nonzero" in k.lower() for k in list(breaks) + list(aten)):
        hints.append("data_dependent_python: S1/S3b candidate (.item / nonzero)")
    if extern and not triton_kernels:
        hints.append("extern_only: vendor GEMM/cuDNN dominates; Inductor mostly launches")
    if triton_kernels and extern:
        hints.append("mixed_triton_extern: fusion happened; leftover in cuBLAS/cuDNN")
    if custom:
        hints.append("custom_kernel_in_graph: replace_pattern landed (gko.* op)")
    extern_names = " ".join(extern)
    triton_names = " ".join(triton_kernels)
    gemm_share = prof_buckets.get("gemm_conv", 0.0)
    adam_share = prof_buckets.get("adam", 0.0)
    if ("addmm" in extern_names or extern.get("mm")) and "gelu" in triton_names.lower():
        hints.append(
            "missed_gemm_epilogue: mm/addmm extern + gelu triton; S6 only if leftover "
            "triton/copy is material (not GEMM-dominated). Writer must fuse epi in-register."
        )
    if gemm_share >= 60:
        hints.append("already_hardware_efficient: >=60% CUDA in GEMM/conv")
    if adam_share >= 10:
        hints.append("adam_launch: S3 compile_optimizer candidate")
    if cache_limit:
        hints.append("recompile_cache_size_limit: too many specializations / hooks")
    if guard_kinds.get("TENSOR_MATCH", 0) >= 20:
        hints.append("many_static_shape_guards: consider mark_dynamic on the varying dim")

    return {
        "raw_tagged_line_counts": dict(tagged_lines),
        "G": {
            "graph_break_from": breaks.most_common(20),
            "user_frames_sample": list(dict.fromkeys(user_frames))[:12],
        },
        "D": {
            "guard_kinds": guard_kinds.most_common(20),
            "unique_graphs": unique_graphs or None,
            "calls_captured": calls_captured or None,
            "cache_size_limit_hits": cache_limit,
        },
        "A": {
            "aten_ops": aten.most_common(25),
            "dtypes": sorted(dtypes),
        },
        "I": {
            "extern_kernels": extern.most_common(15),
            "triton_kernel_names": triton_kernels.most_common(15),
            "custom_kernels": custom.most_common(15),
        },
        "P": {
            "top_kernels": prof_top[:15],
            "cuda_time_share_pct": {k: round(v, 1) for k, v in sorted(prof_buckets.items(), key=lambda x: -x[1])},
        },
        "hints": hints,
        "dropped": [
            "full guard tree / TENSOR_MATCH stride dumps",
            "full AOT FX graph node list",
            "Triton kernel source and inductor boilerplate",
            "profiler events under 1% (empty/copy/record_function)",
        ],
        "compression": {
            "raw_bytes": raw_bytes,
            "tagged_lines": dict(tagged_lines),
        },
    }


def parse_profiler_table(text: str) -> tuple:
    """Best-effort parse of profiler .table() text. CUDA time = last us/ms column when present."""
    rows = []
    for line in text.splitlines():
        if "----" in line or line.strip().startswith("Name") or not line.strip():
            continue
        if "Self CPU" in line or "Self CUDA" in line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        # Last token that looks like a number with unit is noisy; use CUDA total if 11+ cols.
        name_end = None
        for i, p in enumerate(parts):
            if p.endswith("%") and i > 0:
                name_end = i
                break
        if name_end is None:
            continue
        name = " ".join(parts[:name_end]).strip()
        nums = []
        for p in parts[name_end:]:
            try:
                nums.append(float(p.replace("%", "").replace("us", "").replace("ms", "")))
            except ValueError:
                continue
        if len(nums) < 2:
            continue
        # Heuristic: in the 11-col CUDA table, Self CUDA is index ~5 of numeric fields.
        cuda = nums[5] if len(nums) >= 6 else nums[-2]
        rows.append((name, cuda))

    total = sum(c for _, c in rows if c > 0) or 1.0
    buckets: collections.Counter = collections.Counter()
    top = []
    for name, cuda in sorted(rows, key=lambda x: -x[1])[:40]:
        share = 100.0 * cuda / total
        buckets[_bucket_kernel(name)] += share
        if share < 1.0:
            continue
        top.append({"name": name[:80], "cuda_share_pct": round(share, 2)})
    return top, dict(buckets)


def summarize_profiler_obj(prof) -> Dict[str, Any]:
    """In-process torch.profiler.profile → same P slice as parse_profiler_table."""
    ka = prof.key_averages()
    items = []
    use_cuda = True
    try:
        total = sum(float(getattr(e, "self_cuda_time_total", 0) or 0) for e in ka)
        if total <= 0:
            use_cuda = False
            total = sum(float(e.self_cpu_time_total or 0) for e in ka) or 1.0
    except Exception:
        use_cuda = False
        total = 1.0
    buckets: collections.Counter = collections.Counter()
    for e in ka:
        dt = float(getattr(e, "self_cuda_time_total", 0) or 0) if use_cuda else float(e.self_cpu_time_total or 0)
        share = 100.0 * dt / total
        buckets[_bucket_kernel(e.key)] += share
        items.append((e.key, share))
    items.sort(key=lambda x: -x[1])
    top = [{"name": n[:80], "cuda_share_pct": round(s, 2)} for n, s in items if s >= 1.0][:15]
    return {
        "top_kernels": top,
        "cuda_time_share_pct": {k: round(v, 1) for k, v in sorted(buckets.items(), key=lambda x: -x[1])},
        "metric": "self_cuda_time_total" if use_cuda else "self_cpu_time_total",
    }


def format_report(summary: Dict[str, Any]) -> str:
    lines = ["=== filtered feedback (decision slice) ==="]
    for key in ("G", "D", "A", "I", "P"):
        lines.append(f"-- {key} --")
        lines.append(json.dumps(summary.get(key, {}), indent=2, ensure_ascii=False))
    lines.append("-- hints --")
    for h in summary.get("hints") or ["(none)"]:
        lines.append(f"  * {h}")
    lines.append("-- dropped from raw dump --")
    for d in summary.get("dropped") or []:
        lines.append(f"  - {d}")
    rc = summary.get("raw_tagged_line_counts") or {}
    if rc:
        lines.append(f"raw TORCH_LOGS line counts: {rc}")
    comp = summary.get("compression") or {}
    if comp.get("raw_bytes"):
        approx = len(json.dumps(summary, ensure_ascii=False).encode("utf-8"))
        lines.append(f"compression: {comp['raw_bytes']} raw bytes -> ~{approx} summary bytes")
    return "\n".join(lines)


def _selftest() -> Dict[str, Any]:
    import io
    import logging

    import torch
    from torch.profiler import ProfilerActivity, profile

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    loggers = [
        logging.getLogger("torch._dynamo"),
        logging.getLogger("torch._inductor"),
        logging.getLogger("torch._functorch"),
    ]
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
    try:
        import torch._logging as tlog

        if hasattr(tlog, "set_logs"):
            tlog.set_logs(graph_breaks=True, recompiles=True, guards=True, aot_graphs=True, output_code=True)
    except Exception:
        pass

    x = torch.randn(64, 64, device="cuda", requires_grad=True)

    def f(t):
        return (t @ t).relu().sum()

    cf = torch.compile(f, backend="inductor")
    for _ in range(2):
        cf(x).backward()
        x.grad = None
    acts = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "CUDA"):
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, record_shapes=False) as prof:
        cf(x).backward()

    summary = parse_log_text(buf.getvalue())
    summary["P"] = summarize_profiler_obj(prof)
    try:
        from torch._dynamo.utils import counters

        summary["D"]["counters_stats"] = dict(counters.get("stats", {}))
    except Exception:
        pass
    for lg in loggers:
        lg.removeHandler(handler)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", nargs="*", default=[], help="TORCH_LOGS / profiler text dumps")
    p.add_argument("--json", action="store_true", help="print JSON only")
    p.add_argument("--selftest", action="store_true", help="tiny compile+profiler on GPU")
    p.add_argument("-o", "--out", default="", help="write summary JSON")
    args = p.parse_args()
    if not args.log and not args.selftest:
        p.error("pass --log FILE [FILE ...] or --selftest")

    merged: Optional[Dict[str, Any]] = None
    texts = []
    for path in args.log:
        texts.append(Path(path).read_text(errors="replace"))
    if texts:
        merged = parse_log_text("\n".join(texts))
    if args.selftest:
        live = _selftest()
        if merged is None:
            merged = live
        else:
            merged["selftest_P"] = live.get("P")
            merged["hints"] = list(dict.fromkeys((merged.get("hints") or []) + (live.get("hints") or [])))

    assert merged is not None
    if args.out:
        Path(args.out).write_text(json.dumps(merged, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)
    if args.json:
        print(json.dumps(merged, indent=2))
    else:
        print(format_report(merged), flush=True)


if __name__ == "__main__":
    main()
