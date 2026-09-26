"""Per-case compile traces: read-only dumps for G/D/A/I/P.

Does not write into Inductor's cache. Artifacts live under
``run/gko_eval/traces/<case>/`` so retrieval always knows where to look.

Collector dumps vanilla here by default. Recipe ``{"op": "dump_trace"}`` dumps
the *agent* compile into ``traces/<case>/recipe/`` on a one-shot first run
(so timed steps do not keep TORCH_LOGS on).
"""

from __future__ import annotations

import io
import json
import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

GKO = Path(__file__).resolve().parents[1]
TRACE_ROOT = GKO / "run" / "gko_eval" / "traces"
MAX_LOG_BYTES = 4 * 1024 * 1024


def case_key(case: str) -> str:
    return (case or "unknown").replace("/", "__")


def trace_dir_for(case: str, subdir: str = "") -> Path:
    dest = TRACE_ROOT / case_key(case)
    if subdir:
        dest = dest / subdir
    return dest


def dump_trace_dir(recipe: dict) -> Optional[Path]:
    """Where recipe dump_trace should land, or None if the recipe does not dump."""
    if recipe.get("dump_trace") is False:
        return None
    acts = [a for a in recipe.get("actions", []) if a.get("op") == "dump_trace"]
    if not acts and not recipe.get("dump_trace"):
        return None
    act = acts[0] if acts else {}
    case = act.get("case") or recipe.get("model") or "unknown"
    sub = act.get("subdir") or "recipe"
    return Path(act["dir"]) if act.get("dir") else trace_dir_for(case, sub)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False) + "\n")


def write_explain(dest: Path, expl: dict) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "explain.json"
    write_json(path, expl)
    return path


class _Tee:
    def __init__(self, primary, buf: io.StringIO, *, echo: bool = False):
        self.primary = primary
        self.buf = buf
        self.echo = echo

    def write(self, s):
        self.buf.write(s)
        if self.echo:
            try:
                self.primary.write(s)
            except Exception:
                pass
        return len(s) if isinstance(s, str) else 0

    def flush(self):
        try:
            self.primary.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.primary, name)


def _parse_and_write(dest: Path, text: str, *, full: bool, cuda: bool) -> dict:
    from filter_feedback import parse_log_text

    parsed = parse_log_text(text)
    raw_path = dest / "torch_logs.txt"
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) > MAX_LOG_BYTES:
        raw_path.write_bytes(
            encoded[: MAX_LOG_BYTES // 2]
            + b"\n\n...[truncated]...\n\n"
            + encoded[-(MAX_LOG_BYTES // 2) :]
        )
    else:
        raw_path.write_bytes(encoded)
    write_json(dest / "summary.json", parsed)
    write_json(
        dest / "I.json",
        {
            "extern_kernels": (parsed.get("I") or {}).get("extern_kernels"),
            "triton_kernel_names": (parsed.get("I") or {}).get("triton_kernel_names"),
            "custom_kernels": (parsed.get("I") or {}).get("custom_kernels"),
        },
    )
    write_json(dest / "D.json", parsed.get("D") or {})
    write_json(dest / "A.json", parsed.get("A") or {})
    write_json(
        dest / "manifest.json",
        {
            "dir": str(dest),
            "full": full,
            "cuda": cuda,
            "files": sorted(p.name for p in dest.iterdir() if p.is_file()),
            "hints": parsed.get("hints") or [],
        },
    )
    return parsed


@contextmanager
def TraceCapture(dest: Path, *, full: bool = False, echo: bool = False):
    """Write compile traces into dest (not Inductor's cache dir).

    Light mode: inductor output_code files + TORCH_LOGS artifacts (names only).
    full=True: also FX/IR dumps under dest/inductor.
    """
    import sys

    import torch
    import torch._inductor.config as inductor_config

    dest = Path(dest)
    # Wipe leftover inductor output_code from a previous round/run. Parsing
    # stale gko::addmm_gelu files makes identity/S3b look like a fusion land.
    # Never rmtree the traces root itself.
    if dest.exists() and dest.resolve() != TRACE_ROOT.resolve():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    trace = inductor_config.trace
    prev = {
        "enabled": trace.enabled,
        "debug_dir": trace.debug_dir,
        "fx_graph": trace.fx_graph,
        "fx_graph_transformed": trace.fx_graph_transformed,
        "ir_pre_fusion": trace.ir_pre_fusion,
        "ir_post_fusion": trace.ir_post_fusion,
        "output_code": trace.output_code,
        "graph_diagram": trace.graph_diagram,
    }
    cache_prev = {}
    for name in ("force_disable_caches", "fxgraph_cache", "fx_graph_cache"):
        if hasattr(inductor_config, name):
            cache_prev[name] = getattr(inductor_config, name)
    if hasattr(inductor_config, "force_disable_caches"):
        inductor_config.force_disable_caches = True
    for name in ("fxgraph_cache", "fx_graph_cache"):
        if hasattr(inductor_config, name):
            setattr(inductor_config, name, False)
    trace.enabled = True
    trace.debug_dir = str(dest / "inductor")
    trace.output_code = True
    if not full:
        trace.fx_graph = False
        trace.fx_graph_transformed = False
        trace.ir_pre_fusion = False
        trace.ir_post_fusion = False
        trace.graph_diagram = False

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, buf, echo=echo)
    sys.stderr = _Tee(old_err, buf, echo=echo)
    try:
        import torch._logging as tlog

        if hasattr(tlog, "set_logs"):
            tlog.set_logs(
                graph_breaks=True,
                recompiles=True,
                guards=True,
                aot_graphs=True,
                output_code=True,
            )
    except Exception:
        pass

    summary: dict[str, Any] = {}
    try:
        yield summary
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        root_logger.removeHandler(handler)
        for k, v in prev.items():
            setattr(trace, k, v)
        for k, v in cache_prev.items():
            setattr(inductor_config, k, v)
        try:
            import torch._logging as tlog

            if hasattr(tlog, "set_logs"):
                tlog.set_logs()
        except Exception:
            pass

        chunks = [buf.getvalue()]
        inductor_dir = dest / "inductor"
        if inductor_dir.is_dir():
            for py in sorted(inductor_dir.rglob("*.py")):
                try:
                    chunks.append(py.read_text(errors="replace")[:MAX_LOG_BYTES])
                except Exception:
                    pass
        parsed = _parse_and_write(
            dest,
            "\n".join(chunks),
            full=full,
            cuda=bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
        )
        summary.update(parsed)


def capture_first_compile(
    compiled,
    inputs,
    dest: Path,
    *,
    amp: bool = False,
    opt=None,
    step_fn=None,
    full: bool = False,
) -> dict:
    """Force one compiled step under TraceCapture, then leave logs off."""
    import torch
    from torch._dynamo.testing import reduce_to_scalar_loss

    dest = Path(dest)
    with TraceCapture(dest, full=full) as summary:
        ctx = torch.cuda.amp.autocast(enabled=amp and torch.cuda.is_available())
        with ctx:
            if step_fn is not None:
                step_fn(compiled, inputs, opt, amp=amp)
            elif isinstance(inputs, dict):
                pred = compiled(**inputs)
                if torch.is_tensor(pred):
                    pred.sum().backward()
                elif hasattr(pred, "loss") and pred.loss is not None:
                    pred.loss.backward()
                else:
                    try:
                        reduce_to_scalar_loss(pred).backward()
                    except Exception:
                        pass
            else:
                seq = inputs if isinstance(inputs, (list, tuple)) else (inputs,)
                pred = compiled(*seq)
                if torch.is_tensor(pred):
                    pred.sum().backward()
        if hasattr(compiled, "zero_grad"):
            compiled.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return summary


def attach_summary_to_features(features: dict, summary: dict, trace_dir: Path) -> None:
    """Fold compressed I/D/A into Collector observation (not raw Triton)."""
    features["trace_dir"] = str(trace_dir)
    i = summary.get("I") or {}
    d = summary.get("D") or {}
    a = summary.get("A") or {}
    features["inductor_extern"] = i.get("extern_kernels")
    features["inductor_triton"] = i.get("triton_kernel_names")
    features["inductor_custom"] = i.get("custom_kernels")
    features["guard_kinds"] = d.get("guard_kinds")
    features["cache_size_limit_hits"] = d.get("cache_size_limit_hits")
    features["aten_ops"] = a.get("aten_ops")
    features["trace_dtypes"] = a.get("dtypes")
    features["trace_hints"] = summary.get("hints") or []
