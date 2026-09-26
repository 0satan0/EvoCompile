"""Collector: eager + naive torch.compile(model) + inspect features."""

from __future__ import annotations

import inspect
import sys
import traceback

from agent import GKO
from agent.state import Observation, Timing


def _amp_for(name: str, suite: str, no_amp: bool) -> bool:
    if no_amp:
        return False
    amp = suite in ("huggingface", "timm")
    if amp and suite == "huggingface":
        try:
            from common import load_yaml_file

            if name in (load_yaml_file("huggingface.yaml").get("only_fp32") or []):
                return False
        except Exception:
            pass
    return amp


def _model_tree(model, limit: int = 40) -> str:
    lines = [f"root={type(model).__name__}"]
    n = 0
    for name, child in model.named_children():
        lines.append(f"  {name}: {type(child).__name__}")
        n += 1
        if n >= limit:
            lines.append("  ...")
            break
    return "\n".join(lines)


def _forward_src(model, limit: int = 80) -> str:
    fn = getattr(type(model), "forward", None)
    if fn is None:
        return ""
    try:
        src = inspect.getsource(fn)
    except Exception:
        return ""
    lines = src.splitlines()
    if len(lines) > limit:
        src = "\n".join(lines[:limit]) + "\n# ... truncated"
    return src[:4000]


def _eval_imports():
    sys.path.insert(0, str(GKO / "eval"))
    from inspect_features import (  # noqa: E402
        _arch_class,
        _embed_hidden,
        _fwd_explain,
        _input_layout,
        _layerdrop_n,
        _opt_kind,
        _param_stats,
        _prof_buckets,
        _repeat_blocks,
    )
    from run_agent_cases import (  # noqa: E402
        _detect_suite,
        _load,
        _setup_cwd,
        _step_fn,
        _time_loop,
        runtime_step,
    )

    return {
        "arch_class": _arch_class,
        "embed_hidden": _embed_hidden,
        "fwd_explain": _fwd_explain,
        "input_layout": _input_layout,
        "layerdrop_n": _layerdrop_n,
        "opt_kind": _opt_kind,
        "param_stats": _param_stats,
        "prof_buckets": _prof_buckets,
        "repeat_blocks": _repeat_blocks,
        "detect_suite": _detect_suite,
        "load": _load,
        "setup_cwd": _setup_cwd,
        "step_fn": _step_fn,
        "runtime_step": runtime_step,
        "time_loop": _time_loop,
    }


def _time_variant(time_loop, step, warmup: int, measure: int) -> Timing:
    try:
        ms, compile_s, peak, loss = time_loop(step, warmup=warmup, measure=measure)
        return Timing(ok=True, ms=ms, compile_s=compile_s, peak_gib=peak, loss=loss)
    except Exception:
        return Timing(ok=False, err=traceback.format_exc())


class CollectorAgent:
    def __init__(
        self,
        *,
        warmup: int = 8,
        measure: int = 40,
        prof: bool = True,
        no_amp: bool = False,
        dump_trace: bool = True,
        explain: bool = True,
    ):
        self.warmup = warmup
        self.measure = measure
        self.prof = prof
        self.no_amp = no_amp
        self.dump_trace = dump_trace
        self.explain = explain

    def collect(self, case: str, suite: str = "") -> Observation:
        import gc

        import torch

        ev = _eval_imports()
        ev["setup_cwd"]()
        try:
            from kernels import reset_installed

            reset_installed()
        except Exception:
            pass
        suite = ev["detect_suite"](case, suite)
        amp = _amp_for(case, suite, self.no_amp)
        obs = Observation(model=case, suite=suite, amp=amp, opt="unknown")

        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
        _bm, model, inputs, opt = ev["load"](case, suite=suite)
        obs.opt = ev["opt_kind"](opt)
        stats = ev["param_stats"](model)
        blocks = ev["repeat_blocks"](model)
        hidden = ev["embed_hidden"](model)
        obs.features = {
            "model": case,
            "suite": suite,
            "amp": amp,
            "opt": obs.opt,
        "layerdrop_n": ev["layerdrop_n"](model),
        "arch_class": ev["arch_class"](stats, blocks, hidden, stats["conv_param_frac"]),
        "hidden": hidden,
        "blocks": blocks,
        **stats,
        **ev["input_layout"](inputs),
    }
        obs.model_tree = _model_tree(model)
        obs.forward_src = _forward_src(model)

        expl = {}
        if self.explain:
            expl = ev["fwd_explain"](model, inputs, amp)
            obs.features.update(expl)
        step_fn = ev.get("runtime_step", lambda _n: ev["step_fn"])(case)
        time_loop = ev["time_loop"]

        if self.dump_trace:
            from trace_dump import trace_dir_for, write_explain

            write_explain(trace_dir_for(case), expl or {})
            obs.features["trace_dir"] = str(trace_dir_for(case))

        def eager_step():
            return step_fn(model, inputs, opt, amp=amp)

        w = self.warmup
        obs.eager = _time_variant(time_loop, eager_step, w, self.measure)

        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
        _bm, model, inputs, opt = ev["load"](case, suite=suite)

        try:
            cmodel = torch.compile(model, backend="inductor", fullgraph=False)

            def vanilla_step():
                return step_fn(cmodel, inputs, opt, amp=amp)

            obs.vanilla = _time_variant(time_loop, vanilla_step, max(w, 3), self.measure)
        except Exception:
            obs.vanilla = Timing(ok=False, err=traceback.format_exc())

        if self.prof or self.dump_trace:
            try:
                torch._dynamo.reset()
                gc.collect()
                torch.cuda.empty_cache()
                _bm, model, inputs, opt = ev["load"](case, suite=suite)
                if self.dump_trace:
                    from trace_dump import TraceCapture, attach_summary_to_features, trace_dir_for, write_json

                    dest = trace_dir_for(case, "vanilla")
                    with TraceCapture(dest) as summary:
                        if self.prof:
                            obs.features["profiler"] = ev["prof_buckets"](model, inputs, amp)
                        else:
                            cmodel = torch.compile(model, backend="inductor", fullgraph=False)
                            ev["runtime_step"](case)(cmodel, inputs, opt, amp=amp)
                    attach_summary_to_features(obs.features, summary, trace_dir_for(case))
                    if self.prof and "profiler" in obs.features:
                        write_json(trace_dir_for(case) / "profiler.json", obs.features["profiler"])
                elif self.prof:
                    obs.features["profiler"] = ev["prof_buckets"](model, inputs, amp)
            except Exception as e:
                obs.features["profiler_err"] = f"{type(e).__name__}: {e}"
            try:
                from torch._dynamo.utils import counters

                obs.features["dynamo_counters"] = {k: dict(v) for k, v in counters.items() if v}
            except Exception:
                pass

        ratio = obs.speedup_vanilla_over_eager()
        obs.features["vanilla_over_eager"] = None if ratio is None else round(ratio, 3)
        obs.features["eager_ms"] = obs.eager_ms
        obs.features["vanilla_ms"] = obs.vanilla_ms
        obs.features["vanilla_ok"] = obs.vanilla_ok

        # Naive I-channel is compile(model). S6 overlays official compile(train_step),
        # so also dump that leftover; retrieve/repair continue from the wrap in use.
        from agent.fusion_sites import snapshot_i_channel

        channels = dict(obs.features.get("leftover_channels") or {})
        channels["naive"] = snapshot_i_channel(obs.features)
        obs.features["leftover_channels"] = channels
        obs.features["leftover_source"] = "naive"
        if self.dump_trace:
            try:
                torch._dynamo.reset()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _bm, model, inputs, opt = ev["load"](case, suite=suite)
                from agent.residual import features_from_trace_summary
                from trace_dump import TraceCapture, trace_dir_for

                dest = trace_dir_for(case, "official_step")
                body = ev["runtime_step"](case)
                with TraceCapture(dest) as summary:
                    cstep = torch.compile(body, backend="inductor", fullgraph=False)
                    cstep(model, inputs, opt, amp=amp)
                official = features_from_trace_summary(summary)
                if official.get("inductor_extern") or official.get("inductor_triton"):
                    channels["official_step"] = official
                    obs.features["leftover_channels"] = channels
            except Exception as e:
                obs.features["official_leftover_err"] = f"{type(e).__name__}: {e}"
        return obs
