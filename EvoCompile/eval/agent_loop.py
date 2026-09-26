#!/usr/bin/env python3
"""Agent eval loop. Benchmark is not patched per case.

Flow:
  1. Dataset case → load uncompiled model (same TorchBench Model as baseline).
  2. Read frozen baseline from gko_eval.csv (eager, naive compile).
  3. Agent artifact = eval/recipes/<model>.json
     (missing recipe ⇒ identity = naive torch.compile(model); most cases).
  4. apply_recipe(uncompiled_model) → compiled model.
  5. Same train step as baseline → agent_step_ms.
  6. Feedback JSON: beat_naive / beat_eager / error. That is the agent's reward.

Usage:
  python eval/agent_loop.py --case detectron2_maskrcnn_r_50_c4
  python eval/agent_loop.py --case resnet50 --recipe eval/recipes/resnet50.json

Input:  uncompiled model (and baseline numbers).
Output: recipe-applied compiled model + whether it beats naive compile.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

GKO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GKO / "eval"))

from recipe import (  # noqa: E402
    apply_input_ops,
    apply_recipe,
    beat_naive,
    identity_recipe,
    load_recipe,
    recipe_path_for,
    save_recipe,
)
from run_agent_cases import (  # noqa: E402
    _detect_suite,
    _fwd_bwd,
    _load,
    _loss_to_float,
    _opt_zero,
    _setup_cwd,
    _time_loop,
    _upsert,
    is_inference_only,
    runtime_step,
)
from schema import empty_row  # noqa: E402


def _baseline_from_csv(csv_path: Path, model: str) -> dict:
    out = {
        "eager_ms": None,
        "vanilla_ms": None,
        "vanilla_ok": None,
        "source": "",
    }
    if not csv_path.exists():
        return out
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("model") != model:
                continue
            eager = row.get("eager_or_uncompiled_step_ms") or ""
            vanilla = row.get("vanilla_compile_step_ms") or ""
            status = row.get("status") or ""
            if eager:
                try:
                    out["eager_ms"] = float(eager)
                except ValueError:
                    pass
            if vanilla:
                try:
                    out["vanilla_ms"] = float(vanilla)
                    out["vanilla_ok"] = True
                except ValueError:
                    out["vanilla_ok"] = False
            elif "fail" in status and "vanilla" in status:
                out["vanilla_ok"] = False
            out["source"] = row.get("case_id", "")
            if row.get("case_id", "").startswith("agent_"):
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--case",
        required=True,
        help="Model name: TorchBench, HuggingFace (e.g. BertForMaskedLM), or TIMM",
    )
    p.add_argument(
        "--suite",
        default="",
        help="torchbench | huggingface | timm. Empty = auto-detect.",
    )
    p.add_argument("--recipe", default="", help="JSON recipe; default eval/recipes/<case>.json")
    p.add_argument(
        "--variants",
        default="eager,vanilla,agent",
        help="Comma-separated: eager,vanilla,agent,step,step_no_opt. "
        "step = torch.compile(whole train_step including optimizer); "
        "step_no_opt = compile fwd+loss+bwd, optimizer stays eager.",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--measure", type=int, default=20)
    p.add_argument(
        "--baseline-csv",
        default=str(GKO / "run" / "gko_eval" / "gko_eval.csv"),
    )
    p.add_argument(
        "--out-csv",
        default=str(GKO / "run" / "gko_eval" / "gko_eval.csv"),
    )
    p.add_argument(
        "--feedback-dir",
        default=str(GKO / "run" / "gko_eval" / "feedback"),
    )
    p.add_argument(
        "--amp",
        action="store_true",
        default=None,
        help="CUDA AMP (Dynamo HF/TIMM training uses this). Default: on for huggingface/timm.",
    )
    p.add_argument("--no-amp", action="store_true", help="Disable AMP")
    p.add_argument(
        "--write-identity",
        action="store_true",
        help="If no recipe exists, write identity recipe (naive compile) and run it",
    )
    args = p.parse_args()
    args.variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    _setup_cwd()

    name = args.case
    rpath = Path(args.recipe) if args.recipe else recipe_path_for(name)
    if rpath.exists():
        recipe = load_recipe(rpath)
        print(f"recipe: {rpath}", flush=True)
    elif args.write_identity:
        recipe = identity_recipe()
        save_recipe(recipe, recipe_path_for(name))
        print(f"wrote identity recipe {recipe_path_for(name)}", flush=True)
    else:
        recipe = identity_recipe()
        print(
            "no case-specific recipe; using identity = naive torch.compile(model). "
            "Most cases should stay here.",
            flush=True,
        )

    csv_baseline = _baseline_from_csv(Path(args.baseline_csv), name)
    print(f"csv_baseline: {csv_baseline}", flush=True)
    suite = _detect_suite(name, args.suite)
    amp = False if args.no_amp else (True if args.amp else suite in ("huggingface", "timm"))
    # Dynamo huggingface.yaml only_fp32 (e.g. GoogleFnet FFT cannot AMP).
    if amp and not args.amp and suite == "huggingface":
        from common import load_yaml_file

        if name in (load_yaml_file("huggingface.yaml").get("only_fp32") or []):
            amp = False
    print(f"amp={amp} suite={suite}", flush=True)

    import gc
    import traceback

    import torch

    results = {}
    desc = recipe.get("note", "")
    for variant in args.variants:
        print(f"== {name} {variant} ==", flush=True)
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
        try:
            bm, model, inputs, opt = _load(name, suite=args.suite)
            body = runtime_step(name)
            if variant == "eager":

                def step():
                    return body(model, inputs, opt, amp=amp)

                w = args.warmup
            elif variant == "vanilla":
                cmodel = torch.compile(model, backend="inductor", fullgraph=False)

                def step():
                    return body(cmodel, inputs, opt, amp=amp)

                w = max(args.warmup, 3)
            elif variant == "step":
                # Training: Dynamo compile(train_step). Inference: compile(forward_pass).
                cstep = torch.compile(body, backend="inductor", fullgraph=False)

                def step():
                    return cstep(model, inputs, opt, amp=amp)

                w = max(args.warmup, 3)
            elif variant == "step_no_opt":
                if is_inference_only(name) or opt is None:
                    raise RuntimeError("step_no_opt is a training wrap; this case is inference-only")
                cbody = torch.compile(
                    _fwd_bwd, backend="inductor", fullgraph=False
                )

                def step():
                    _opt_zero(opt)
                    loss = cbody(model, inputs, amp=amp)
                    opt.step()
                    return _loss_to_float(loss)

                w = max(args.warmup, 3)
            elif variant == "agent":
                from contextlib import nullcontext

                from trace_dump import TraceCapture, dump_trace_dir

                dump_dir = dump_trace_dir(recipe)
                dump_ctx = (
                    TraceCapture(
                        dump_dir,
                        full=bool(recipe.get("dump_trace_full")),
                    )
                    if dump_dir is not None
                    else nullcontext()
                )
                with dump_ctx:
                    inputs, in_notes = apply_input_ops(inputs, recipe)
                    compiled, desc = apply_recipe(model, recipe)
                    if in_notes:
                        desc = f"{'; '.join(in_notes)}; {desc}"
                    if recipe.get("compile_step"):
                        cstep = torch.compile(
                            body, backend="inductor", fullgraph=False
                        )
                        desc = f"{desc}; compile_step"
                        print(f"  applied: {desc}", flush=True)

                        def step():
                            return cstep(compiled, inputs, opt, amp=amp)

                        w = max(args.warmup, 3)
                    else:
                        if recipe.get("compile_optimizer") and opt is not None:
                            body(compiled, inputs, opt, amp=amp)
                            opt.step = torch.compile(opt.step, backend="inductor")
                            desc = f"{desc}; compile_optimizer"
                        print(f"  applied: {desc}", flush=True)

                        def step():
                            return body(compiled, inputs, opt, amp=amp)

                        w = max(args.warmup, 3)
                    if dump_dir is not None:
                        step()
            else:
                raise ValueError(variant)
            ms, compile_s, peak, loss = _time_loop(
                step, warmup=w, measure=args.measure
            )
            results[variant] = {
                "ok": True,
                "ms": ms,
                "compile_s": compile_s,
                "peak": peak,
                "loss": loss,
                "err": "",
            }
            print(
                f"  {variant}: {ms:.2f} ms  compile={compile_s:.1f}s  "
                f"peak={peak:.2f}GiB  loss={loss}",
                flush=True,
            )
        except Exception:
            err = traceback.format_exc()
            print(err, flush=True)
            results[variant] = {
                "ok": False,
                "ms": None,
                "compile_s": None,
                "peak": None,
                "loss": None,
                "err": err,
            }
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    eager = results.get("eager") or {}
    vanilla = results.get("vanilla") or {}
    agent = results.get("agent") or {}
    eager_ms = eager.get("ms")
    vanilla_ms = vanilla.get("ms")
    vanilla_ok = vanilla.get("ok") if "vanilla" in results else csv_baseline.get("vanilla_ok")
    if vanilla_ms is None and csv_baseline.get("vanilla_ms") is not None:
        vanilla_ms = csv_baseline["vanilla_ms"]
    if eager_ms is None and csv_baseline.get("eager_ms") is not None:
        eager_ms = csv_baseline["eager_ms"]
    ms = agent.get("ms")
    compile_s = agent.get("compile_s")
    peak = agent.get("peak")
    loss = agent.get("loss")
    agent_ok = bool(agent.get("ok"))
    err = agent.get("err") or ""

    verdict = beat_naive(eager_ms, vanilla_ms, vanilla_ok, ms, agent_ok)
    feedback = {
        "model": name,
        "recipe_path": str(rpath),
        "recipe": recipe,
        "applied": desc,
        "csv_baseline": csv_baseline,
        "measured": {
            k: {kk: vv for kk, vv in v.items() if kk != "err"}
            for k, v in results.items()
        },
        "agent_step_ms": ms,
        "compile_s": compile_s,
        "peak_gib": peak,
        "loss": loss,
        "error": err[-2000:] if err else "",
        **verdict,
    }
    fbdir = Path(args.feedback_dir)
    fbdir.mkdir(parents=True, exist_ok=True)
    fbpath = fbdir / f"{name.replace('/', '__')}.json"
    fbpath.write_text(json.dumps(feedback, indent=2, default=str) + "\n")
    print(
        f"feedback: {fbpath} beat_naive={verdict['beat_naive']} {verdict['reason']}",
        flush=True,
    )

    row = empty_row(
        case_id=f"agent_{name.replace('/', '__')}",
        suite=suite,
        model=name,
        task="inference" if is_inference_only(name) else "training",
        gpu_count="1",
        parallelism="none",
        compile_config=desc if agent_ok else "agent_fail",
        eager_or_uncompiled_step_ms=f"{eager_ms:.4f}" if eager_ms is not None else "",
        vanilla_compile_step_ms=(
            f"{vanilla_ms:.4f}" if vanilla_ok and vanilla_ms is not None else ""
        ),
        agent_step_ms=f"{ms:.4f}" if ms is not None else "",
        compile_time=f"{compile_s:.4f}" if compile_s is not None else "",
        peak_memory=f"{peak:.4f}" if peak is not None else "",
        status="pass:agent" if verdict["beat_naive"] else ("pass" if agent_ok else "fail:agent"),
    )
    _upsert(Path(args.out_csv), row)


if __name__ == "__main__":
    main()
