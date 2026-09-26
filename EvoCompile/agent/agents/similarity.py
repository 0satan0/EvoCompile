"""Similarity supplement: two-channel kNN on compact trials.

Wins suggest transfer. Losses ban accel schemes on similar fingerprints.
Hard stop-leaves (F2-skinny / F6 / …) are kept — they ARE the failure lesson.
The LLM never sees source, recipes, or the full trial log.
"""

from __future__ import annotations

from typing import Optional

from agent.casememory import attach_casememory
from agent.fingerprint import case_card, compact_fp
from agent.llm import LLMClient, parse_json_object
from agent.memory_tree import ACCEL_SCHEMES, MemoryStore, STOP_SCHEMES
from agent.prompts import SYSTEM_SIMILARITY, user_similarity
from agent.state import Observation, SchemeHit

KNN_K = 5
DIST_TRANSFER = 0.32
IDENTITY_LIKE = {"identity"}


class SimilarityAgent:
    def __init__(self, store: MemoryStore, llm: Optional[LLMClient] = None, *, k: int = KNN_K):
        self.store = store
        self.llm = llm
        self.k = k

    def neighbors(self, obs: Observation, *, k: int | None = None, polarity: str = "all"):
        fp = compact_fp(obs)
        return self.store.neighbors(
            fp, polarity=polarity, k=k or self.k, exclude_case=obs.model
        )

    def annotate(self, obs: Observation, hard: SchemeHit) -> SchemeHit:
        fp = compact_fp(obs)
        wins = self.store.neighbors(fp, polarity="win", k=self.k, exclude_case=obs.model)
        losses = self.store.neighbors(fp, polarity="lose", k=self.k, exclude_case=obs.model)
        hard.similar = [case_card(e, dist=d) for d, e in wins]
        hard.similar_losses = [case_card(e, dist=d) for d, e in losses]
        hard.banned = self.store.banned_schemes(fp, exclude_case=obs.model)
        return hard

    def merge(
        self,
        obs: Observation,
        hard: SchemeHit,
        *,
        enabled: bool = True,
    ) -> SchemeHit:
        hard = self.annotate(obs, hard)
        if not enabled:
            return hard
        if hard.name in STOP_SCHEMES and hard.node_id != "identity-fallback":
            return hard
        if hard.name in ACCEL_SCHEMES and hard.name in hard.banned:
            # Nearby loss says this accel already failed on a similar fp — do not re-probe it.
            hard.reason = (
                f"{hard.reason} [banned by similar loss; skip {hard.name}]"
            )
            hard.confidence = "low"
            return hard
        if hard.name != "identity":
            return hard
        if hard.node_id == "identity-micro":
            return hard
        transfer = self._decide(obs, hard)
        if not transfer:
            return hard
        scheme = transfer.get("scheme")
        if not scheme or scheme not in self.store.schemes():
            return hard
        if scheme in hard.banned:
            return hard
        reason = (
            f"hard miss ({hard.reason}); similarity from "
            f"{transfer.get('from_case') or 'knn'}: {transfer.get('why') or scheme}"
        )
        hit = self.store.hit_from_scheme(scheme, obs.model, reason=reason, source="similarity")
        hit.similar = hard.similar
        hit.similar_losses = hard.similar_losses
        hit.banned = hard.banned
        hit.path = list(hard.path)
        hit.confidence = "medium"
        attach_casememory(hit)
        return hit

    def _decide(self, obs: Observation, hard: SchemeHit) -> Optional[dict]:
        cards = [c for c in hard.similar if c.get("scheme") not in set(hard.banned)]
        if not cards:
            return None
        if self.llm is None:
            return self._knn_heuristic(cards)
        fp = compact_fp(obs)
        names = sorted(
            {c.get("scheme") for c in (hard.similar + hard.similar_losses) if c.get("scheme")}
        )
        one = self.store.scheme_one_liners(names)
        user = user_similarity(
            current=fp,
            hard=hard.name,
            neighbors=hard.similar,
            losses=hard.similar_losses,
            banned=hard.banned,
            schemes=one,
        )
        raw = self.llm.chat(SYSTEM_SIMILARITY, user)
        obj = parse_json_object(raw) or {}
        if not obj.get("transfer"):
            return None
        return obj

    def _knn_heuristic(self, cards: list[dict]) -> Optional[dict]:
        winners = [c for c in cards if c.get("verdict") == "win" or c.get("beat_naive")]
        pool = winners or [c for c in cards if c.get("scheme") in self.store.schemes()]
        if not pool:
            return None
        best = pool[0]
        dist = float(best.get("dist") or 1.0)
        if dist > DIST_TRANSFER:
            return None
        if best.get("scheme") in IDENTITY_LIKE:
            return None
        return {
            "transfer": True,
            "scheme": best["scheme"],
            "from_case": best.get("case"),
            "why": f"knn dist={dist:.3f}",
        }
