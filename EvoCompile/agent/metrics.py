"""Ablation / paper stats: success is run+numerics, speedup uses every case.

Success = the optimized program still executes, and its scalar loss matches
the pre-optimize (eager) loss under torch.allclose (atol=rtol=1e-2).
It is NOT agent_ms < 0.95 * vanilla_ms, and not a 5% relative-error gate.

Arithmetic and geometric mean speedups include every case. A crash or
numerical mismatch does not drop the case: deployed time falls back to
vanilla (or eager), so that case contributes 1.0× vs the fallback baseline.
"""

from __future__ import annotations

import math
from typing import Any, Optional

# Same rule as torch.allclose(candidate, baseline, atol=atol, rtol=rtol):
#   abs(candidate - baseline) <= atol + rtol * abs(baseline)
LOSS_ATOL = 1e-2
LOSS_RTOL = 1e-2
LOSS_REL_TOL = LOSS_RTOL  # back-compat name; not a 5% relative-only gate


def as_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def loss_consistent(
    baseline: Any,
    candidate: Any,
    *,
    atol: float = LOSS_ATOL,
    rtol: float = LOSS_RTOL,
    rel: Optional[float] = None,
) -> Optional[bool]:
    """None if either loss is missing; else torch.allclose vs eager baseline.

    Matches ``torch.allclose(candidate, baseline, atol=atol, rtol=rtol)`` on
    scalars. ``rel=`` is accepted as an alias for ``rtol`` (old callers).
    """
    if rel is not None:
        rtol = rel
    b = as_float(baseline)
    c = as_float(candidate)
    if b is None or c is None:
        return None
    return abs(c - b) <= atol + rtol * abs(b)


def is_deployable(
    ok: Any = None,
    correctness_ok: Any = None,
    *,
    result: Any = None,
) -> bool:
    """Runnable and not known-wrong. Missing correctness check still counts."""
    if result is not None:
        if isinstance(result, dict):
            ok = result.get("ok")
            correctness_ok = result.get("correctness_ok")
        else:
            ok = getattr(result, "ok", ok)
            correctness_ok = getattr(result, "correctness_ok", correctness_ok)
    if not ok:
        return False
    if correctness_ok is False:
        return False
    return True


def speedup(baseline: Any, deployed: Any) -> float:
    """baseline / deployed. Missing or non-positive times → 1.0 so the case stays in the mean."""
    b = as_float(baseline)
    d = as_float(deployed)
    if b is None or d is None or b <= 0 or d <= 0:
        return 1.0
    return b / d


def arith_mean(xs: list[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def geo_mean(xs: list[float]) -> Optional[float]:
    if not xs:
        return None
    logs = []
    for x in xs:
        if x is None or x <= 0:
            return None
        logs.append(math.log(x))
    return math.exp(sum(logs) / len(logs))


def deployed_ms(
    *,
    eager_ms: Any,
    vanilla_ms: Any,
    agent_ms: Any,
    success: bool,
) -> Optional[float]:
    """Time you would ship: successful agent, else vanilla, else eager."""
    if success:
        ms = as_float(agent_ms)
        if ms is not None:
            return ms
    v = as_float(vanilla_ms)
    if v is not None:
        return v
    return as_float(eager_ms)


def case_speedups(
    *,
    eager_ms: Any,
    vanilla_ms: Any,
    agent_ms: Any,
    success: bool,
) -> dict:
    deployed = deployed_ms(
        eager_ms=eager_ms,
        vanilla_ms=vanilla_ms,
        agent_ms=agent_ms,
        success=success,
    )
    return {
        "success": bool(success),
        "deployed_ms": deployed,
        "speedup_vs_eager": speedup(eager_ms, deployed),
        "speedup_vs_vanilla": speedup(vanilla_ms, deployed),
    }


def _round_eval(rec: dict) -> dict:
    ev = rec.get("eval") if isinstance(rec, dict) else None
    return ev if isinstance(ev, dict) else {}


def _best_deployable(
    rounds: list, eager_loss: Any = None
) -> tuple[Optional[dict], Optional[dict]]:
    best_ev = None
    best_rec = None
    for rec in rounds or []:
        if not isinstance(rec, dict):
            continue
        ev = _round_eval(rec)
        cons = ev.get("correctness_ok")
        if cons is None:
            cons = loss_consistent(eager_loss, ev.get("loss"))
        if not is_deployable(ok=ev.get("ok"), correctness_ok=cons):
            continue
        ms = as_float(ev.get("ms"))
        if ms is None:
            continue
        if best_ev is None or ms < (as_float(best_ev.get("ms")) or math.inf):
            patched = dict(ev)
            if patched.get("correctness_ok") is None:
                patched["correctness_ok"] = cons
            best_ev, best_rec = patched, rec
    return best_rec, best_ev


def case_from_summary(obj: dict, observation: Optional[dict] = None) -> dict:
    """Normalize one case (old heuristic run or new LLM run) into metric fields."""
    observation = observation or {}
    eager = observation.get("eager") or {}
    status = obj.get("status")
    eager_ms = as_float(obj.get("eager_ms"))
    if eager_ms is None:
        eager_ms = as_float(eager.get("ms"))
    vanilla_ms = as_float(obj.get("vanilla_ms"))
    if vanilla_ms is None:
        vanilla_ms = as_float((observation.get("vanilla") or {}).get("ms"))
    eager_loss = eager.get("loss")
    rounds = obj.get("rounds") or []
    _best_rec, best_ev = _best_deployable(rounds, eager_loss)
    correctness_ok = None
    if best_ev is not None:
        agent_ms = as_float(best_ev.get("ms"))
        correctness_ok = best_ev.get("correctness_ok")
        if correctness_ok is None:
            correctness_ok = loss_consistent(eager_loss, best_ev.get("loss"))
        success = is_deployable(ok=True, correctness_ok=correctness_ok)
    elif rounds:
        agent_ms = None
        success = False
    else:
        agent_ms = as_float(obj.get("best_agent_ms"))
        success = bool(
            agent_ms is not None and status not in ("eager_failed", "loop_error")
        )

    if status in ("eager_failed", "loop_error"):
        success = False
        agent_ms = None

    speed = case_speedups(
        eager_ms=eager_ms,
        vanilla_ms=vanilla_ms,
        agent_ms=agent_ms,
        success=success,
    )
    return {
        "case": obj.get("case") or observation.get("model"),
        "suite": obj.get("suite") or observation.get("suite"),
        "status": status,
        "eager_ms": eager_ms,
        "vanilla_ms": vanilla_ms,
        "best_agent_ms": agent_ms,
        "correctness_ok": None if best_ev is None else best_ev.get("correctness_ok", correctness_ok),
        "beat_naive": obj.get("beat_naive"),
        **speed,
    }


def _mean_block(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "arith": None if not xs else round(arith_mean(xs), 6),
        "geo": None if not xs else round(geo_mean(xs), 6),
    }


def aggregate_cases(rows: list[dict]) -> dict:
    """Success rate and speedup means over ALL rows (no 5% filter)."""
    n = len(rows)
    n_success = sum(1 for r in rows if r.get("success"))
    vs_e = [float(r.get("speedup_vs_eager") or 1.0) for r in rows]
    vs_v = [float(r.get("speedup_vs_vanilla") or 1.0) for r in rows]
    by_suite: dict[str, list] = {}
    for r in rows:
        by_suite.setdefault(r.get("suite") or "?", []).append(r)

    def suite_block(bag: list[dict]) -> dict:
        return {
            "n": len(bag),
            "success": sum(1 for r in bag if r.get("success")),
            "success_rate": None if not bag else round(sum(1 for r in bag if r.get("success")) / len(bag), 4),
            "vs_eager": _mean_block([float(r.get("speedup_vs_eager") or 1.0) for r in bag]),
            "vs_vanilla": _mean_block([float(r.get("speedup_vs_vanilla") or 1.0) for r in bag]),
        }

    return {
        "n": n,
        "success": n_success,
        "success_rate": None if not n else round(n_success / n, 4),
        "success_definition": "ran without error and loss matches eager under torch.allclose(atol=1e-2, rtol=1e-2); not a speedup gate",
        "speedup_definition": "all cases; crash/mismatch falls back to vanilla then eager (ratio 1.0 vs that baseline)",
        "vs_eager": _mean_block(vs_e),
        "vs_vanilla": _mean_block(vs_v),
        "by_suite": {k: suite_block(v) for k, v in sorted(by_suite.items())},
        "cases": rows,
    }
