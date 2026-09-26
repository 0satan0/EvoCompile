#!/usr/bin/env python3
"""Compile multi-agent loop.

  collect → hard tree → win/lose kNN → recipe → eval, for --max-rounds iterations.
  A run that *succeeds* re-reads leftover G/I keys and composes a complementary
  scheme onto the working recipe (depth). A miss or crash jumps to another
  unused scheme on the frozen original Observation (breadth).
  beat_naive is recorded; residual depth runs unless --stop-on-win.
  Scheme usage examples live in memory/casememory and are attached at retrieve time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_GKO = Path(__file__).resolve().parents[1]
if str(_GKO) not in sys.path:
    sys.path.insert(0, str(_GKO))

from agent import GKO
from agent.agents.evolver import EvolveAgent
from agent.agents.generator import OptimizerAgent
from agent.agents.repairer import (
    RepairAgent,
    failed_intervention_layer,
    preserve_kernel_landing,
)
from agent.agents.retriever import RetrieverAgent
from agent.agents.similarity import SimilarityAgent
from agent.agents.rewrite_writer import (
    RewriteWriterAgent,
    inject_rewrite_id,
    is_rewrite_error,
    recipe_wants_generated_rewrite,
)
from agent.agents.triton_writer import (
    CATALOG_KERNEL_IDS,
    TritonWriterAgent,
    catalog_fallback,
    inject_kernel_id,
    is_triton_error,
    kernel_id_from_recipe,
    recipe_wants_generated_kernel,
)
from agent.casememory import CaseMemory, attach_casememory, load_casememory
from agent.fusion_sites import sites_from_features
from agent.llm import LLMClient, LLMUnreachable, LLM_DOWN_EXIT
from agent.memory_tree import ACCEL_SCHEMES, obs_case_entry
from agent.residual import (
    compose_recipes,
    depth_hit,
    fusion_missed,
    leftover_fusion_hole,
    observation_for_s6,
    overlay_observation,
)
from agent.metrics import case_speedups, is_deployable
from agent.state import EvalResult, Observation, RoundRecord, SchemeHit

TRITON_REPAIR_CAP = int(os.environ.get("GKO_TRITON_REPAIR_CAP", "5"))
REWRITE_REPAIR_CAP = int(os.environ.get("GKO_REWRITE_REPAIR_CAP", "5"))
RECIPE_REPAIR_CAP = int(os.environ.get("GKO_RECIPE_REPAIR_CAP", "5"))


def _error_fingerprint(err: str, n: int = 400) -> str:
    lines = [ln.strip() for ln in (err or "").splitlines() if ln.strip()]
    tail = lines[-1] if lines else (err or "").strip()
    return tail[:n]


def _recipe_aggression(recipe: dict | None) -> list[str]:
    rec = recipe or {}
    tags: list[str] = []
    if rec.get("compile_optimizer"):
        tags.append("compile_optimizer=true")
    if rec.get("compile_step"):
        tags.append("compile_step=true")
    if rec.get("fullgraph"):
        tags.append("fullgraph=true")
    for a in rec.get("actions") or []:
        op = a.get("op")
        if op == "replace_pattern":
            tags.append(f"custom kernel replace_pattern kernel={a.get('kernel')}")
        elif op == "apply_rewrite":
            tags.append(f"apply_rewrite rewrite={a.get('rewrite')}")
        elif op == "mark_dynamic":
            tags.append("mark_dynamic")
    return tags


def mismatch_error(obs: Observation, result: EvalResult, recipe: dict | None) -> str:
    """Pack rel_loss into a repair prompt: ran, but output drifted from eager."""
    eager_loss = None if obs.eager is None else getattr(obs.eager, "loss", None)
    tags = _recipe_aggression(recipe)
    via = (" via " + "; ".join(tags)) if tags else ""
    return (
        "NUMERICS: fail:rel_loss. The optimized program ran, but its output (loss) "
        "does not match the original eager program"
        f"{via}. eager_loss={eager_loss} agent_loss={getattr(result, 'loss', None)} "
        f"correctness={getattr(result, 'correctness', None)} "
        "allclose atol=1e-2 rtol=1e-2 (vs eager). "
        "This is a failure of THIS landing, not a signal to abandon the strategy. "
        "Keep the intended change (compiled Adam, custom kernel, rewrite, compile "
        "region) and fix the implementation so numerics match eager. "
        "STRICTLY FORBIDDEN: remove replace_pattern / apply_rewrite / the custom "
        "kernel just so allclose passes. "
        "The orchestrator already stores any earlier correct recipe as best — "
        "do not copy that; repair this attempt."
    )


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False) + "\n")


def _recipes_equal(a: dict, b: dict) -> bool:
    def norm(r):
        r = dict(r or {})
        r.pop("note", None)
        return json.dumps(r, sort_keys=True, default=str)

    return norm(a) == norm(b)


def _experience_extra(scheme: SchemeHit) -> str:
    chunks = []
    if scheme.case_cards:
        chunks.append(
            "SCHEME USAGE EXAMPLES (expert experience from OTHER cases — "
            "USAGE ILLUSTRATIONS only; adapt to THIS observation; keep THIS case name):\n"
            + json.dumps(scheme.case_cards[:3], indent=2, default=str, ensure_ascii=False)[:5000]
        )
    if scheme.similar:
        chunks.append(
            "NEAR WINS (compact):\n"
            + json.dumps(scheme.similar[:4], indent=2, default=str, ensure_ascii=False)[:1600]
        )
    if scheme.similar_losses:
        chunks.append(
            "NEAR LOSSES — do not re-try these schemes:\n"
            + json.dumps(scheme.similar_losses[:4], indent=2, default=str, ensure_ascii=False)[:1600]
        )
    if scheme.banned:
        chunks.append("BANNED SCHEMES: " + json.dumps(scheme.banned))
    return ("\n" + "\n".join(chunks)) if chunks else ""


def run_case(
    case: str,
    *,
    suite: str = "",
    max_rounds: int = 3,
    warmup: int = 8,
    measure: int = 40,
    prof: bool = True,
    no_amp: bool = False,
    llm: LLMClient | None = None,
    out_dir: Path | None = None,
    save_recipe: bool = False,
    use_similarity: bool = True,
    use_evolve: bool = True,
    write_memory: bool = True,
    stop_on_win: bool = False,
) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or (GKO / "run" / "gko_eval" / "agent_runs" / case.replace("/", "__") / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    from agent.agents.collector import CollectorAgent
    from agent.agents.evaluator import EvaluatorAgent

    collector = CollectorAgent(warmup=warmup, measure=measure, prof=prof, no_amp=no_amp)
    retriever = RetrieverAgent()
    similarity = SimilarityAgent(retriever.store, llm)
    evolver = EvolveAgent(retriever.store, llm)
    optimizer = OptimizerAgent(llm)
    repairer = RepairAgent(llm)
    triton_writer = TritonWriterAgent(llm)
    rewrite_writer = RewriteWriterAgent(llm)
    evaluator = EvaluatorAgent(warmup=warmup, measure=measure)

    print(f"== collect {case} ==", flush=True)
    obs: Observation = collector.collect(case, suite=suite)
    _dump(out_dir / "observation.json", obs.to_dict())
    print(
        f"  eager={obs.eager_ms} vanilla={obs.vanilla_ms} ok={obs.vanilla_ok} "
        f"opt={obs.opt} arch={obs.features.get('arch_class')} "
        f"breaks={obs.features.get('fwd_breaks')}",
        flush=True,
    )
    if not obs.eager.ok:
        summary = {
            "case": case,
            "status": "eager_failed",
            "error": obs.eager.err[-2000:],
            "out_dir": str(out_dir),
        }
        _dump(out_dir / "summary.json", summary)
        print("eager failed; stopping (loader, not compile)", flush=True)
        return summary

    print("== retrieve (hard tree) ==", flush=True)
    hard: SchemeHit = retriever.retrieve(obs)
    attach_casememory(hard)
    print(f"  hard={hard.name} [{hard.node_id}]: {hard.reason}", flush=True)
    if hard.case_cards:
        print(
            f"  casememory={[c.get('id') for c in hard.case_cards]}",
            flush=True,
        )
    scheme: SchemeHit = similarity.merge(obs, hard, enabled=use_similarity)
    attach_casememory(scheme)
    if scheme.source == "similarity":
        print(f"  similarity → {scheme.name}: {scheme.reason}", flush=True)
    if scheme.similar_losses:
        print(
            f"  losses={[(c.get('case'), c.get('scheme'), c.get('vs_naive_pct')) for c in scheme.similar_losses[:3]]}",
            flush=True,
        )
    tried: list[str] = []
    if (
        use_evolve
        and scheme.name in ACCEL_SCHEMES
        and scheme.name in (scheme.banned or [])
    ):
        print(f"  skip {scheme.name} (banned by similar loss)", flush=True)
        tried.append(scheme.name)
        alt = evolver.explore_next(obs, scheme, tried=tried, banned=scheme.banned)
        if alt is not None:
            scheme = alt
            attach_casememory(scheme)
            print(f"  explore → {scheme.name}: {scheme.reason}", flush=True)
    _dump(
        out_dir / "retrieval.json",
        {
            "hard": hard.to_dict(),
            "chosen": scheme.to_dict(),
            "banned": scheme.banned,
        },
    )
    print(f"  scheme={scheme.name} source={scheme.source}", flush=True)

    rounds: list[RoundRecord] = []
    last_error = ""
    last_recipe: dict = dict(scheme.recipe_hint)
    best: EvalResult | None = None
    best_recipe: dict | None = None
    best_scheme_name = ""
    best_scheme_source = None
    best_scheme_reason = None
    best_idx: int | None = None
    evolve_events: list[dict] = []
    pending_repair = False
    pending_triton = False
    pending_rewrite = False
    pending_catalog = False
    triton_repairs = 0
    rewrite_repairs = 0
    recipe_repairs = 0
    repair_history: list[str] = []
    last_kernel_id = ""
    last_kernel_src = ""
    last_rewrite_id = ""
    last_rewrite_src = ""
    outcomes: list[dict] = []
    prefix_recipe: dict | None = None

    for i in range(max_rounds):
        if scheme is None:
            break
        if use_evolve and not pending_repair and not pending_triton and not pending_rewrite and not pending_catalog and scheme.name in tried:
            prefix_recipe = None
            nxt = evolver.explore_next(
                obs, scheme, tried=tried, banned=scheme.banned
            )
            if nxt is None:
                print("  no unused scheme left; refine current", flush=True)
            else:
                scheme = nxt
                print(f"  breadth → {scheme.name}: {scheme.reason}", flush=True)
                last_recipe = dict(scheme.recipe_hint)
                attach_casememory(scheme)
                triton_repairs = 0
                rewrite_repairs = 0
                recipe_repairs = 0
                repair_history = []

        if last_error:
            if pending_triton:
                mode = "triton_repair"
                print(
                    f"== round {i} triton_repair (same scheme {scheme.name} "
                    f"{triton_repairs}/{TRITON_REPAIR_CAP}) ==",
                    flush=True,
                )
                kid, src, raw = triton_writer.repair(
                    obs,
                    kernel_id=last_kernel_id or "g_fuse",
                    source=last_kernel_src,
                    error=last_error,
                    history=repair_history,
                )
                recipe = inject_kernel_id(last_recipe, kid)
                last_kernel_id, last_kernel_src = kid, src
                pending_triton = False
            elif pending_rewrite:
                mode = "rewrite_repair"
                print(
                    f"== round {i} rewrite_repair (same scheme {scheme.name} "
                    f"{rewrite_repairs}/{REWRITE_REPAIR_CAP}) ==",
                    flush=True,
                )
                rid, src, raw = rewrite_writer.repair(
                    obs,
                    rewrite_id=last_rewrite_id or "g_s1",
                    source=last_rewrite_src,
                    error=last_error,
                    history=repair_history,
                )
                recipe = inject_rewrite_id(last_recipe, rid)
                last_rewrite_id, last_rewrite_src = rid, src
                pending_rewrite = False
            else:
                mode = "repair"
                print(
                    f"== round {i} repair (same scheme {scheme.name} "
                    f"{recipe_repairs}/{RECIPE_REPAIR_CAP}) ==",
                    flush=True,
                )
                layer = failed_intervention_layer(last_recipe, last_error)
                print(f"  repair layer={layer}", flush=True)
                recipe, raw = repairer.repair(
                    obs,
                    last_recipe,
                    last_error,
                    history=repair_history,
                    layer=layer,
                )
                from agent.memory_tree import normalize_s6_recipe

                recipe = preserve_kernel_landing(last_recipe, recipe)
                recipe = normalize_s6_recipe(recipe, scheme.name, obs.model)
                recipe = preserve_kernel_landing(last_recipe, recipe)
                pending_repair = False
                if layer == "C" or recipe_wants_generated_kernel(recipe, scheme.name):
                    view = observation_for_s6(obs)
                    if last_kernel_src and last_kernel_id:
                        kid, src, wraw = triton_writer.repair(
                            view,
                            kernel_id=last_kernel_id,
                            source=last_kernel_src,
                            error=last_error,
                            history=repair_history,
                        )
                    else:
                        kid, src, wraw = triton_writer.generate(
                            view, recipe=recipe, case_tag=f"{obs.model}_{scheme.name}_{i}"
                        )
                    recipe = inject_kernel_id(recipe, kid)
                    last_kernel_id, last_kernel_src = kid, src
                    raw = (raw or "") + "\n--- triton_writer ---\n" + wraw
                    print(f"  triton_writer kernel={kid} (layer C)", flush=True)
                if recipe_wants_generated_rewrite(recipe, scheme.name):
                    rid, src, wraw = rewrite_writer.generate(
                        obs, recipe=recipe, case_tag=f"{obs.model}_{scheme.name}_{i}"
                    )
                    recipe = inject_rewrite_id(recipe, rid)
                    last_rewrite_id, last_rewrite_src = rid, src
                    raw = (raw or "") + "\n--- rewrite_writer ---\n" + wraw
        elif pending_catalog:
            mode = "s6_catalog"
            print(
                f"== round {i} s6_catalog (fusion miss → {last_kernel_id}) ==",
                flush=True,
            )
            recipe = inject_kernel_id(last_recipe, last_kernel_id)
            raw = f"catalog {last_kernel_id} after fusion miss"
            pending_catalog = False
        else:
            mode = "optimize"
            if best_recipe is not None:
                last_recipe = dict(best_recipe)
            extra = _experience_extra(scheme)
            extra = (
                f"This run continues for {max_rounds} rounds (stop_on_win={stop_on_win}). "
                f"Beating naive does not end the loop; use remaining rounds to refine or try a linked example.\n"
                + extra
            )
            if best_recipe is not None and best is not None:
                extra = (
                    f"BEST CORRECT RECIPE is round {best_idx} "
                    f"({best.ms} ms, correctness ok). Iterate from THIS recipe. "
                    "A later slower or crashed round is discarded; do not continue from it.\n"
                    f"{json.dumps(best_recipe, indent=2, default=str, ensure_ascii=False)[:4000]}\n"
                    + extra
                )
            elif rounds and rounds[-1].eval and is_deployable(result=rounds[-1].eval):
                extra = (
                    f"Previous recipe ran: {rounds[-1].eval.ms} ms vs vanilla {obs.vanilla_ms} "
                    f"(beat_naive={rounds[-1].eval.beat_naive} "
                    f"correctness={rounds[-1].eval.correctness}). "
                    f"Now on {scheme.name} ({scheme.source}). Do not repeat tried={tried}.\n"
                    + extra
                )
            if scheme.source == "depth" and prefix_recipe is not None:
                extra = (
                    "DEPTH: a working prefix already ran on the original case. "
                    "Emit complementary actions for this leftover scheme only; "
                    "the orchestrator will compose them onto the prefix. "
                    "Do not drop compile_optimizer / rewrites already in the prefix.\n"
                    + extra
                )
            print(f"== round {i} {mode} ({scheme.name}/{scheme.source}) ==", flush=True)
            recipe, raw = optimizer.generate(obs, scheme, extra=extra)
            from agent.memory_tree import normalize_s6_recipe

            recipe = normalize_s6_recipe(recipe, scheme.name, obs.model)
            if prefix_recipe is not None and scheme.source == "depth":
                recipe = compose_recipes(prefix_recipe, recipe)
                recipe = normalize_s6_recipe(recipe, scheme.name, obs.model)
                print("  compose onto working prefix", flush=True)
            if recipe_wants_generated_kernel(recipe, scheme.name):
                view = observation_for_s6(obs)
                anchor_ev = None
                if best_idx is not None and 0 <= best_idx < len(rounds):
                    rec_b = rounds[best_idx]
                    if rec_b.eval and rec_b.eval.ok:
                        anchor_ev = rec_b.eval
                if (
                    anchor_ev is None
                    and rounds
                    and rounds[-1].eval
                    and rounds[-1].eval.ok
                ):
                    anchor_ev = rounds[-1].eval
                if anchor_ev is not None:
                    view = overlay_observation(obs, anchor_ev.residual)
                kid, src, wraw = triton_writer.generate(
                    view, recipe=recipe, case_tag=f"{obs.model}_{scheme.name}_{i}"
                )
                recipe = inject_kernel_id(recipe, kid)
                last_kernel_id, last_kernel_src = kid, src
                raw = (raw or "") + "\n--- triton_writer ---\n" + wraw
                print(f"  triton_writer kernel={kid}", flush=True)
            if recipe_wants_generated_rewrite(recipe, scheme.name):
                rid, src, wraw = rewrite_writer.generate(
                    obs, recipe=recipe, case_tag=f"{obs.model}_{scheme.name}_{i}"
                )
                recipe = inject_rewrite_id(recipe, rid)
                last_rewrite_id, last_rewrite_src = rid, src
                raw = (raw or "") + "\n--- rewrite_writer ---\n" + wraw
                print(f"  rewrite_writer rewrite={rid}", flush=True)

        if i > 0 and _recipes_equal(recipe, last_recipe) and not last_error and mode != "s6_catalog":
            print("  identical recipe; skip re-eval", flush=True)
            if scheme.name not in tried:
                tried.append(scheme.name)
            if use_evolve:
                nxt = evolver.explore_next(obs, scheme, tried=tried, banned=scheme.banned)
                if nxt is None:
                    continue
                scheme = nxt
                attach_casememory(scheme)
                last_error = ""
                pending_repair = False
                pending_triton = False
                pending_rewrite = False
                triton_repairs = 0
                rewrite_repairs = 0
                recipe_repairs = 0
                repair_history = []
                print(f"  explore → {scheme.name}: {scheme.reason}", flush=True)
            continue
        last_recipe = recipe
        _dump(out_dir / "rounds" / f"{i:02d}_{mode}_recipe.json", recipe)

        print("== eval ==", flush=True)
        result = evaluator.evaluate(obs, recipe)
        rec = RoundRecord(idx=i, mode=mode, recipe=recipe, eval=result, llm_raw=raw)
        rounds.append(rec)
        _dump(out_dir / "rounds" / f"{i:02d}_eval.json", rec.to_dict())

        trial = retriever.store.record_trial(
            obs_case_entry(
                obs,
                scheme=scheme.name,
                result=result.to_dict(),
                source=scheme.source,
            ),
            write=write_memory,
        )
        outcomes.append(
            {
                "scheme": scheme.name,
                "verdict": trial.get("verdict"),
                "vs_naive_pct": result.vs_naive_pct,
                "ok": result.ok,
            }
        )

        if result.ok and not is_deployable(result=result):
            print(
                f"  NUMERICS {result.correctness} agent={result.ms} vs naive "
                f"{result.vs_naive_pct} (keep best round {best_idx})",
                flush=True,
            )
            last_error = mismatch_error(obs, result, recipe)
            fp = _error_fingerprint(last_error)
            if fp:
                repair_history.append(fp)
            pending_catalog = False
            if recipe_repairs < RECIPE_REPAIR_CAP:
                pending_repair = True
                pending_triton = False
                pending_rewrite = False
                recipe_repairs += 1
                print(
                    f"  rel_loss → RepairAgent ({recipe_repairs}/{RECIPE_REPAIR_CAP}); "
                    "fix this landing, do not revert",
                    flush=True,
                )
                continue
            print("  rel_loss repair cap; keep earlier best if any", flush=True)
            pending_repair = False
            last_error = ""
            if scheme.name not in tried:
                tried.append(scheme.name)
            view = obs
            if rounds and rounds[-1].eval:
                view = overlay_observation(obs, rounds[-1].eval.residual)
            if use_evolve:
                nxt = evolver.explore_next(
                    view, scheme, tried=tried, banned=scheme.banned
                )
                if nxt is not None:
                    scheme = nxt
                    attach_casememory(scheme)
                    last_recipe = dict(scheme.recipe_hint)
                    triton_repairs = 0
                    rewrite_repairs = 0
                    recipe_repairs = 0
                    repair_history = []
                    print(f"  explore → {scheme.name}: {scheme.reason}", flush=True)
                    continue
            if best_recipe is not None:
                last_recipe = dict(best_recipe)
            continue

        if result.ok:
            print(
                f"  agent={result.ms:.2f} ms  vs naive {result.vs_naive_pct} "
                f"verdict={trial.get('verdict')} beat_naive={result.beat_naive} {result.reason}",
                flush=True,
            )
            last_error = ""
            pending_repair = False
            pending_triton = False
            pending_rewrite = False
            pending_catalog = False
            if scheme.name not in tried:
                tried.append(scheme.name)
            is_best_round = False
            if is_deployable(result=result) and (
                best is None
                or (result.ms is not None and (best.ms is None or result.ms < best.ms))
            ):
                best, best_recipe, best_scheme_name = result, recipe, scheme.name
                best_scheme_source = scheme.source
                best_scheme_reason = scheme.reason
                best_idx = i
                is_best_round = True
                print(f"  new best round {i} ({result.ms:.2f} ms)", flush=True)
            elif (
                best is not None
                and result.ms is not None
                and best.ms is not None
                and abs(result.ms - best.ms) <= 1e-6
            ):
                is_best_round = True
            else:
                print(
                    f"  keep best round {best_idx} ({None if best is None else best.ms} ms); "
                    "this round is slower — next refine continues from the best",
                    flush=True,
                )
                if best_recipe is not None:
                    last_recipe = dict(best_recipe)
            if not is_best_round:
                continue
            if result.beat_naive and use_evolve:
                evo = evolver.promote(
                    obs, hard, scheme, result, write=write_memory
                )
                evolve_events.append(evo)
                if evo.get("applied"):
                    print(f"  promote → tree leaf {evo.get('new_id')}", flush=True)
                elif str(evo.get("why") or "").startswith("write=False"):
                    print(f"  promote skipped (no-memory-write): {evo.get('new_id')}", flush=True)
            residual = dict(result.residual or {})
            _dump(out_dir / "rounds" / f"{i:02d}_residual.json", residual)
            if fusion_missed(residual, recipe):
                kid = kernel_id_from_recipe(recipe)
                # fused_sdpa lands as fused_sdpa_pair/bwd. Do not swap a landed
                # FA / catalog kernel to addmm_gelu because leftover still has gelu.
                if kid.lower() in CATALOG_KERNEL_IDS:
                    print(
                        f"  S6 catalog {kid} did not rewrite the graph "
                        "(FX matcher missed this IR).",
                        flush=True,
                    )
                else:
                    fb = catalog_fallback(
                        sites_from_features(
                            overlay_observation(obs, residual).features
                        )
                    )
                    if fb and kid != fb:
                        print(
                            f"  S6 did not land custom op (kernel={kid or '?'}); "
                            f"retry catalog {fb}",
                            flush=True,
                        )
                        last_kernel_id = fb
                        pending_catalog = True
                        pending_repair = False
                        pending_triton = False
                        pending_rewrite = False
                        continue
                    print(
                        "  S6 catalog did not rewrite the graph "
                        "(FX matcher missed this IR).",
                        flush=True,
                    )
            nxt_depth = None
            if not stop_on_win:
                nxt_depth = depth_hit(
                    retriever.store,
                    obs,
                    recipe,
                    residual=residual,
                    applied=tried,
                )
            if nxt_depth is not None:
                prefix_recipe = best_recipe or recipe
                scheme = nxt_depth
                attach_casememory(scheme)
                print(f"  depth → {scheme.name}: {scheme.reason}", flush=True)
                triton_repairs = 0
                rewrite_repairs = 0
                recipe_repairs = 0
                repair_history = []
                continue
            if result.beat_naive:
                if leftover_fusion_hole(obs, residual, recipe):
                    print(
                        "  win but fusion hole remains; keep best, try unused schemes",
                        flush=True,
                    )
                else:
                    print("  win and residual closed; keep best", flush=True)
                    break
            if use_evolve and scheme.source == "hard":
                evo = evolver.consider(
                    obs, scheme, result, enabled=True, write=write_memory
                )
                evolve_events.append(evo)
                if evo.get("applied"):
                    print(f"  evolve split {evo.get('new_id')}", flush=True)
                elif str(evo.get("why") or "").startswith("write=False"):
                    print(f"  evolve skipped (no-memory-write): {evo.get('new_id')}", flush=True)
            if not use_evolve:
                continue
            prefix_recipe = None
            nxt = evolver.explore_next(
                obs, scheme, tried=tried, last_result=result, banned=scheme.banned
            )
            if nxt is None:
                print("  no unused scheme; stay and refine", flush=True)
                continue
            scheme = nxt
            attach_casememory(scheme)
            print(f"  breadth → {scheme.name}: {scheme.reason}", flush=True)
            triton_repairs = 0
            rewrite_repairs = 0
            recipe_repairs = 0
            repair_history = []
            continue

        print(f"  FAIL {result.err[-400:]}", flush=True)
        prefix_recipe = None
        pending_catalog = False
        fp = _error_fingerprint(result.err)
        if fp:
            repair_history.append(fp)
        if is_triton_error(result.err, recipe) and triton_repairs < TRITON_REPAIR_CAP:
            pending_triton = True
            pending_rewrite = False
            pending_repair = False
            last_error = result.err
            triton_repairs += 1
            print(
                f"  triton crash → TritonWriter ({triton_repairs}/{TRITON_REPAIR_CAP})",
                flush=True,
            )
            continue
        if is_rewrite_error(result.err, recipe) and rewrite_repairs < REWRITE_REPAIR_CAP:
            pending_rewrite = True
            pending_triton = False
            pending_repair = False
            last_error = result.err
            rewrite_repairs += 1
            print(
                f"  rewrite crash → RewriteWriter ({rewrite_repairs}/{REWRITE_REPAIR_CAP})",
                flush=True,
            )
            continue
        if recipe_repairs < RECIPE_REPAIR_CAP:
            pending_repair = True
            pending_triton = False
            pending_rewrite = False
            last_error = result.err
            recipe_repairs += 1
            print(
                f"  eval crash → RepairAgent ({recipe_repairs}/{RECIPE_REPAIR_CAP})",
                flush=True,
            )
            continue
        if scheme.name not in tried:
            tried.append(scheme.name)
        pending_repair = False
        pending_triton = False
        pending_rewrite = False
        last_error = ""
        triton_repairs = 0
        rewrite_repairs = 0
        recipe_repairs = 0
        repair_history = []
        if best_recipe is not None:
            last_recipe = dict(best_recipe)
        if not use_evolve:
            continue
        nxt = evolver.explore_next(
            obs, scheme, tried=tried, last_result=result, banned=scheme.banned
        )
        if nxt is None:
            continue
        scheme = nxt
        attach_casememory(scheme)
        print(f"  breadth after crash → {scheme.name}", flush=True)

    from agent.fingerprint import compact_fp

    retriever.store.record_path(
        case=obs.model,
        fp=compact_fp(obs),
        tried=tried,
        outcomes=outcomes,
        best=best_scheme_name,
        best_vs_naive_pct=None if best is None else best.vs_naive_pct,
        write=write_memory,
    )

    if write_memory and best_recipe is not None:
        try:
            load_casememory.cache_clear()
            how = (
                f"Collected from this run. scheme={best_scheme_name}. "
                f"beat_naive={bool(best and best.beat_naive)} "
                f"vs_naive_pct={None if best is None else best.vs_naive_pct}. "
                f"Copy the recipe shape for scheme {best_scheme_name}; substitute the current model name."
            )
            path = CaseMemory.load().write_collected(
                scheme=best_scheme_name,
                case=obs.model,
                suite=obs.suite,
                recipe=best_recipe,
                how=how,
                vs_naive_pct=None if best is None else best.vs_naive_pct,
                beat_naive=bool(best and best.beat_naive),
            )
            print(f"  casememory wrote {path.name}", flush=True)
            load_casememory.cache_clear()
        except Exception as e:
            print(f"  casememory write skipped: {e}", flush=True)

    if best_recipe is not None:
        _dump(out_dir / "final_recipe.json", best_recipe)
        if save_recipe:
            sys.path.insert(0, str(GKO / "eval"))
            from recipe import recipe_path_for, save_recipe as _save

            dest = recipe_path_for(case)
            _save(best_recipe, dest)
            print(f"wrote {dest}", flush=True)

    summary = {
        "case": case,
        "suite": obs.suite,
        "scheme": best_scheme_name or (None if scheme is None else scheme.name),
        "scheme_source": best_scheme_source if best_scheme_name else (None if scheme is None else scheme.source),
        "scheme_reason": best_scheme_reason if best_scheme_name else (None if scheme is None else scheme.reason),
        "tried": tried,
        "outcomes": outcomes,
        "eager_ms": obs.eager_ms,
        "vanilla_ms": obs.vanilla_ms,
        "vanilla_ok": obs.vanilla_ok,
        "best_agent_ms": None if best is None else best.ms,
        "best_round": best_idx,
        "success": bool(best and is_deployable(result=best)),
        "correctness": None if best is None else best.correctness,
        "correctness_ok": None if best is None else best.correctness_ok,
        "beat_naive": bool(best and best.beat_naive),
        "beat_eager": bool(best and best.beat_eager),
        "reason": None if best is None else best.reason,
        "rounds": [r.to_dict() for r in rounds],
        "evolve": evolve_events,
        "out_dir": str(out_dir),
        "used_llm": llm is not None,
        "used_similarity": use_similarity,
        "used_evolve": use_evolve,
        "llm_model": None if llm is None else llm.model,
        "llm_usage": None if llm is None else llm.usage_dict(),
    }
    summary.update(
        case_speedups(
            eager_ms=obs.eager_ms,
            vanilla_ms=obs.vanilla_ms,
            agent_ms=None if best is None else best.ms,
            success=bool(best and is_deployable(result=best)),
        )
    )
    _dump(out_dir / "summary.json", summary)
    if llm is not None:
        u = llm.usage_dict()
        print(
            f"  llm_usage calls={u['calls']} prompt={u['prompt_tokens']} "
            f"completion={u['completion_tokens']} total={u['total_tokens']} "
            f"cached={u['cached_tokens']}",
            flush=True,
        )
    print(f"summary: {out_dir / 'summary.json'}", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--suite", default="")
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--measure", type=int, default=40)
    p.add_argument("--no-prof", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-llm", action="store_true", help="Use retrieved recipe_hint, no LLM")
    p.add_argument("--no-similarity", action="store_true", help="Hard tree only")
    p.add_argument("--no-evolve", action="store_true", help="No tree split/promote and no explore transitions")
    p.add_argument("--no-memory-write", action="store_true", help="Do not update treememory or casememory")
    p.add_argument(
        "--stop-on-win",
        action="store_true",
        help="Stop at first beat_naive without residual depth compose (default: deepen on leftover keys, then stop)",
    )
    p.add_argument("--llm-url", default="http://localhost:8000/v1")
    p.add_argument("--llm-model", default="default")
    p.add_argument(
        "--llm-api-key",
        default="",
        help="OpenAI-compatible API key; else DEEPSEEK_API_KEY / OPENAI_API_KEY",
    )
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--out-dir", default="")
    p.add_argument(
        "--save-recipe",
        action="store_true",
        help="Write the best recipe to eval/recipes/<case>.json",
    )
    args = p.parse_args()
    llm = None
    if not args.no_llm:
        llm = LLMClient(
            model=args.llm_model,
            server_url=args.llm_url,
            api_key=args.llm_api_key,
            temperature=args.temperature,
        )
    try:
        run_case(
            args.case,
            suite=args.suite,
            max_rounds=args.max_rounds,
            warmup=args.warmup,
            measure=args.measure,
            prof=not args.no_prof,
            no_amp=args.no_amp,
            llm=llm,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            save_recipe=args.save_recipe,
            use_similarity=not args.no_similarity,
            use_evolve=not args.no_evolve,
            write_memory=not args.no_memory_write,
            stop_on_win=args.stop_on_win,
        )
    except LLMUnreachable as e:
        dest = Path(args.out_dir) if args.out_dir else GKO / "run" / "gko_eval" / "LLM_DOWN.json"
        dest = dest / "LLM_DOWN.json" if dest.suffix != ".json" else dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(
                {
                    "reason": "local_llm_unreachable",
                    "case": args.case,
                    "error": str(e),
                    "when": datetime.now().isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"LLM_DOWN {e} recorded {dest}", flush=True)
        raise SystemExit(LLM_DOWN_EXIT)


if __name__ == "__main__":
    main()
