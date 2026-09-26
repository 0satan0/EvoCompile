"""Evolve the retrieval tree from a failed accel hit.

The LLM sees: current compact fingerprint, the failed leaf's `when`, a few
prototype cards that MUST keep their scheme. It does not see source or the
whole tree. Python validates the proposed `when` before writing treememory.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from agent.casememory import attach_casememory
from agent.fingerprint import case_card, compact_fp, context_from_fp, match_context
from agent.llm import LLMClient, parse_json_object
from agent.memory_tree import ACCEL_SCHEMES, MemoryStore, STOP_SCHEMES, eval_when
from agent.prompts import SYSTEM_EVOLVE, SYSTEM_EXPLORE, user_evolve, user_explore
from agent.state import EvalResult, Observation, SchemeHit

MAX_PROTOTYPES = 6


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s)[:40].strip("_") or "case"


class EvolveAgent:
    def __init__(self, store: MemoryStore, llm: Optional[LLMClient] = None):
        self.store = store
        self.llm = llm

    def consider(
        self,
        obs: Observation,
        scheme: SchemeHit,
        result: EvalResult,
        *,
        enabled: bool = True,
        write: bool = True,
    ) -> dict:
        """Return an event dict. May insert a sibling exception leaf."""
        event = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "case": obs.model,
            "scheme": scheme.name,
            "node_id": scheme.node_id,
            "beat_naive": result.beat_naive,
            "ok": result.ok,
            "applied": False,
            "why": "",
        }
        if not enabled:
            event["why"] = "disabled"
            return event
        if not result.ok:
            event["why"] = "eval failed; repair first, do not split the tree"
            return event
        if result.beat_naive:
            event["why"] = "scheme worked"
            return event
        if scheme.source != "hard":
            event["why"] = "only split on hard-match mistakes"
            return event
        if scheme.name not in ACCEL_SCHEMES:
            event["why"] = f"{scheme.name} is not an accel leaf"
            return event
        if not scheme.node_id:
            event["why"] = "missing node_id"
            return event
        if self.llm is None:
            event["why"] = "no LLM; recorded case only"
            return event

        prototypes = self._prototypes(scheme.name, exclude=obs.model)
        proto_cards = [case_card(p) for p in prototypes[:MAX_PROTOTYPES]]
        failed_leaf = self.store.find_node(scheme.node_id) or {}
        user = user_evolve(
            current=compact_fp(obs),
            failed_scheme=scheme.name,
            failed_when=failed_leaf.get("when") or {},
            path=scheme.path,
            vs_naive_pct=result.vs_naive_pct,
            prototypes=proto_cards,
            allowed_schemes=sorted(self.store.schemes().keys()),
        )
        raw = self.llm.chat(SYSTEM_EVOLVE, user)
        event["llm_raw"] = (raw or "")[:2000]
        obj = parse_json_object(raw) or {}
        if not obj.get("apply"):
            event["why"] = obj.get("reason") or "LLM declined split"
            return event
        when = obj.get("when")
        new_scheme = obj.get("scheme")
        if not isinstance(when, dict) or not new_scheme:
            event["why"] = "malformed when/scheme"
            return event
        if new_scheme not in self.store.schemes():
            event["why"] = f"unknown scheme {new_scheme}"
            return event
        if new_scheme == scheme.name:
            event["why"] = "split must change scheme"
            return event

        if not eval_when(when, _ctx(obs)):
            event["why"] = "proposed when does not match the failed case"
            return event
        for p in prototypes:
            if eval_when(when, context_from_fp(p.get("fp") or p)):
                event["why"] = f"proposed when would steal prototype {p.get('case')}"
                return event

        new_id = f"evo_{_slug(obs.model)}_{_slug(new_scheme)}"
        if self.store.find_node(new_id):
            new_id = new_id + "_" + datetime.now().strftime("%H%M%S")
        catalog = self.store.schemes().get(new_scheme) or {}
        new_node = {
            "id": new_id,
            "kind": "leaf",
            "when": when,
            "scheme": new_scheme,
            "reason": obj.get("reason") or f"evolve split from {scheme.name} after {obs.model}",
            "recipe": catalog.get("recipe") or {"kind": "identity", "note": new_scheme},
            "do_not": list(catalog.get("do_not") or []),
            "examples": obs.model,
            "confidence": "medium",
            "origin": "evolve",
            "created_from": scheme.node_id,
        }

        cards = [c for c in self.store.cases() if c.get("case") != obs.model]
        if not self._routes_preserved(cards, new_node, scheme.node_id):
            event["why"] = "dry-run: would reroute a historical case"
            return event

        event["new_id"] = new_id
        event["new_scheme"] = new_scheme
        event["when"] = when
        if not write:
            event["applied"] = False
            event["why"] = "write=False; " + new_node["reason"]
            return event
        if not self.store.insert_before(scheme.node_id, new_node):
            event["why"] = "insert failed"
            return event
        event["applied"] = True
        event["why"] = new_node["reason"]
        self.store.log_evolve(event, write=True)
        return event

    def _prototypes(self, scheme: str, *, exclude: str) -> list[dict]:
        out = []
        for c in self.store.cases():
            if c.get("case") == exclude:
                continue
            if c.get("scheme") == scheme and c.get("beat_naive"):
                out.append(c)
        return out

    def _routes_preserved(self, cards: list[dict], new_node: dict, sibling_id: str) -> bool:
        """Insert in a copy of the tree; every historical card must keep its current route."""
        import copy

        before: dict[str, str] = {}
        for card in cards:
            case = card.get("case")
            if not case:
                continue
            ctx = context_from_fp(card.get("fp") or card)
            hit, _ = self.store.walk_ctx(ctx, model=case)
            before[case] = hit.name
        backup = copy.deepcopy(self.store.data)
        try:
            if not self.store.insert_before(sibling_id, new_node):
                return False
            for card in cards:
                case = card.get("case")
                if not case:
                    continue
                ctx = context_from_fp(card.get("fp") or card)
                hit, _ = self.store.walk_ctx(ctx, model=case)
                if hit.name != before[case]:
                    return False
            return True
        finally:
            self.store.data = backup

    def candidates(
        self,
        obs: Observation,
        current: str,
        *,
        tried: list[str],
        banned: list[str],
    ) -> list[str]:
        table = self.store.explore_from()
        raw = list(table.get(current) or [])
        ctx = match_context(obs)
        blocked = set(tried) | set(banned) | {current}
        out: list[str] = []
        for name in raw:
            if name in blocked:
                continue
            if name not in self.store.schemes():
                continue
            if not _legal_on(obs, name, ctx):
                continue
            out.append(name)
        return out

    def explore_next(
        self,
        obs: Observation,
        current: SchemeHit,
        *,
        tried: list[str],
        last_result: EvalResult | None = None,
        banned: list[str] | None = None,
    ) -> SchemeHit | None:
        """Pick a not-yet-tried scheme. None = stop exploring."""
        if current.name in STOP_SCHEMES and current.node_id != "identity-fallback":
            if current.name not in ("F2-cnn", "F2-skinny"):
                return None
        banned = list(banned if banned is not None else current.banned)
        cands = self.candidates(obs, current.name, tried=tried, banned=banned)
        if not cands:
            return None
        chosen = cands[0]
        why = f"explore transition {current.name} → {chosen}"
        if self.llm is not None:
            fp = compact_fp(obs)
            wins = [case_card(e, dist=d) for d, e in self.store.neighbors(fp, polarity="win", k=4, exclude_case=obs.model)]
            losses = [case_card(e, dist=d) for d, e in self.store.neighbors(fp, polarity="lose", k=4, exclude_case=obs.model)]
            vs = None if last_result is None else last_result.vs_naive_pct
            verdict = "none"
            if last_result is not None:
                verdict = "crash" if not last_result.ok else (
                    "win" if last_result.beat_naive else "miss"
                )
            raw = self.llm.chat(
                SYSTEM_EXPLORE,
                user_explore(
                    current=fp,
                    current_scheme=current.name,
                    last_verdict=verdict,
                    vs_naive_pct=vs,
                    tried=tried,
                    candidates=cands,
                    banned=banned,
                    wins=wins,
                    losses=losses,
                ),
            )
            obj = parse_json_object(raw) or {}
            nxt = obj.get("next")
            if nxt == "stop":
                return None
            if nxt in cands:
                chosen = nxt
                why = obj.get("why") or why
            elif nxt == "identity" and "identity" in cands:
                chosen = "identity"
                why = obj.get("why") or why
        hit = self.store.hit_from_scheme(
            chosen, obs.model, reason=why, source="explore"
        )
        hit.similar = current.similar
        hit.similar_losses = current.similar_losses
        hit.banned = banned
        hit.path = list(current.path)
        attach_casememory(hit)
        return hit

    def promote(
        self,
        obs: Observation,
        hard: SchemeHit,
        winner: SchemeHit,
        result: EvalResult,
        *,
        write: bool = True,
    ) -> dict:
        """On a large win that is not the hard leaf, insert a narrow path update."""
        event = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "case": obs.model,
            "kind": "promote",
            "from": hard.name,
            "to": winner.name,
            "applied": False,
            "why": "",
        }
        if not result.beat_naive:
            event["why"] = "not a large win"
            return event
        if winner.name == hard.name and winner.source != "explore":
            event["why"] = "hard leaf already matches the win"
            return event
        sibling = hard.node_id or "identity-fallback"
        if not self.store.find_node(sibling):
            sibling = "identity-fallback"
        when = {"name_contains": [obs.model]}
        new_id = f"evo_{_slug(obs.model)}_{_slug(winner.name)}"
        if self.store.find_node(new_id):
            new_id = new_id + "_" + datetime.now().strftime("%H%M%S")
        catalog = self.store.schemes().get(winner.name) or {}
        new_node = {
            "id": new_id,
            "kind": "leaf",
            "when": when,
            "scheme": winner.name,
            "reason": (
                f"promoted {winner.name} after {obs.model} "
                f"vs_naive={result.vs_naive_pct} (was {hard.name})"
            ),
            "recipe": catalog.get("recipe") or {"kind": "identity", "note": winner.name},
            "do_not": list(catalog.get("do_not") or []),
            "examples": obs.model,
            "confidence": "medium",
            "origin": "promote",
            "created_from": sibling,
        }
        cards = [c for c in self.store.cases() if c.get("case") != obs.model]
        if not self._routes_preserved(cards, new_node, sibling):
            event["why"] = "dry-run: would reroute a historical case"
            return event
        event["new_id"] = new_id
        event["when"] = when
        if not write:
            event["applied"] = False
            event["why"] = "write=False; " + new_node["reason"]
            return event
        if not self.store.insert_before(sibling, new_node):
            event["why"] = "insert failed"
            return event
        event["applied"] = True
        event["why"] = new_node["reason"]
        self.store.log_evolve(event, write=True)
        return event


def _legal_on(obs: Observation, scheme: str, ctx: dict) -> bool:
    if scheme == "S2" and int(ctx.get("layerdrop_n") or 0) < 1:
        return False
    if scheme.startswith("S3") or scheme == "S5+S3":
        if not ctx.get("adam"):
            return False
    if scheme == "S3-fc" and float(ctx.get("conv") or 0) >= 0.3 and float(ctx.get("sync") or 0) < 40:
        return False
    if scheme == "S1" and not ctx.get("hot_breaks"):
        return False
    if scheme == "S2" and not ctx.get("adam"):
        return False
    if scheme == "S6-fuse" and not ctx.get("has_unfused_epilogue"):
        return False
    if scheme == "S6-sdpa" and not ctx.get("unfused_bmm_attention"):
        return False
    _ = obs
    return True


def _ctx(obs: Observation) -> dict:
    from agent.fingerprint import match_context

    return match_context(obs)
