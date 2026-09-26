"""Evaluator: apply_recipe + time the train step vs collected eager/vanilla."""

from __future__ import annotations

import gc
import sys
import traceback
from contextlib import nullcontext

from agent import GKO
from agent.metrics import loss_consistent
from agent.state import EvalResult, Observation

sys.path.insert(0, str(GKO / "eval"))

from recipe import apply_input_ops, apply_recipe, beat_naive  # noqa: E402
from run_agent_cases import _load, _setup_cwd, _time_loop, runtime_step  # noqa: E402


class EvaluatorAgent:
    def __init__(self, *, warmup: int = 8, measure: int = 40):
        self.warmup = warmup
        self.measure = measure

    def evaluate(self, obs: Observation, recipe: dict) -> EvalResult:
        import torch

        from agent.residual import features_from_trace_summary

        _setup_cwd()
        try:
            from kernels import reset_installed

            reset_installed()
        except Exception:
            pass
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
        _bm, model, inputs, opt = _load(obs.model, suite=obs.suite)
        amp = obs.amp
        w = max(self.warmup, 3)
        residual: dict = {}
        body = runtime_step(obs.model)
        try:
            dump_ctx = nullcontext()
            dump_dir = None
            try:
                from trace_dump import TraceCapture, dump_trace_dir, trace_dir_for

                dump_dir = dump_trace_dir(recipe)
                if dump_dir is None:
                    dump_dir = trace_dir_for(obs.model, "residual")
                dump_act = next(
                    (a for a in recipe.get("actions", []) if a.get("op") == "dump_trace"),
                    {},
                )
                dump_ctx = TraceCapture(
                    dump_dir,
                    full=bool(dump_act.get("full") or recipe.get("dump_trace_full")),
                )
            except Exception:
                dump_ctx = nullcontext()

            captured = None
            with dump_ctx as captured:
                inputs, in_notes = apply_input_ops(inputs, recipe)
                compiled, desc = apply_recipe(model, recipe)
                if in_notes:
                    desc = f"{'; '.join(in_notes)}; {desc}"
                if recipe.get("compile_step"):
                    cstep = torch.compile(body, backend="inductor", fullgraph=False)
                    desc = f"{desc}; compile_step"

                    def step():
                        return cstep(compiled, inputs, opt, amp=amp)

                else:
                    if recipe.get("compile_optimizer") and opt is not None:
                        body(compiled, inputs, opt, amp=amp)
                        opt.step = torch.compile(opt.step, backend="inductor")
                        desc = f"{desc}; compile_optimizer"

                    def step():
                        return body(compiled, inputs, opt, amp=amp)

                step()
            if isinstance(captured, dict):
                residual.update(features_from_trace_summary(captured))

            ms, compile_s, peak, loss = _time_loop(step, warmup=w, measure=self.measure)
            verdict = beat_naive(
                obs.eager_ms, obs.vanilla_ms, obs.vanilla_ok, ms, True
            )
            vs = None
            if obs.vanilla_ms and ms:
                vs = (obs.vanilla_ms - ms) / obs.vanilla_ms * 100.0
            cons = loss_consistent(obs.eager.loss, loss)  # allclose atol=rtol=1e-2 vs eager
            if cons is True:
                cnote = "pass"
            elif cons is False:
                cnote = "fail:rel_loss"
            else:
                cnote = "n/a"
            reason = verdict["reason"]
            if cons is False:
                reason = f"{reason}; output mismatch vs eager loss"
            return EvalResult(
                ok=True,
                ms=ms,
                compile_s=compile_s,
                peak_gib=peak,
                loss=loss,
                applied=desc,
                beat_naive=bool(verdict["beat_naive"]),
                beat_eager=bool(verdict["beat_eager"]),
                reason=reason,
                vs_naive_pct=vs,
                residual=residual,
                correctness_ok=cons,
                correctness=cnote,
            )
        except Exception:
            err = traceback.format_exc()
            return EvalResult(
                ok=False,
                err=err,
                reason="agent failed to run",
                correctness_ok=False,
                correctness="fail:crash",
            )
