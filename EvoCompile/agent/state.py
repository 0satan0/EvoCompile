"""Shared dataclasses for the compile multi-agent loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Timing:
    ok: bool
    ms: Optional[float] = None
    compile_s: Optional[float] = None
    peak_gib: Optional[float] = None
    loss: Any = None
    err: str = ""


@dataclass
class Observation:
    """Step 1: eager + naive compile + inspect features."""

    model: str
    suite: str
    amp: bool
    opt: str
    features: dict = field(default_factory=dict)
    eager: Timing = field(default_factory=lambda: Timing(ok=False))
    vanilla: Timing = field(default_factory=lambda: Timing(ok=False))
    model_tree: str = ""
    forward_src: str = ""

    @property
    def eager_ms(self) -> Optional[float]:
        return self.eager.ms if self.eager.ok else None

    @property
    def vanilla_ms(self) -> Optional[float]:
        return self.vanilla.ms if self.vanilla.ok else None

    @property
    def vanilla_ok(self) -> Optional[bool]:
        if self.vanilla.err and not self.vanilla.ok:
            return False
        if self.vanilla.ok:
            return True
        return None

    def speedup_vanilla_over_eager(self) -> Optional[float]:
        e, v = self.eager_ms, self.vanilla_ms
        if e and v and v > 0:
            return e / v
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class SchemeHit:
    name: str
    reason: str
    recipe_hint: dict
    do_not: list = field(default_factory=list)
    examples: str = ""
    case_cards: list = field(default_factory=list)
    source: str = "hard"  # hard | similarity | evolve | explore | depth
    confidence: str = "hard"  # hard | medium | low
    node_id: str = ""
    path: list = field(default_factory=list)
    similar: list = field(default_factory=list)
    similar_losses: list = field(default_factory=list)
    banned: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalResult:
    ok: bool
    ms: Optional[float] = None
    compile_s: Optional[float] = None
    peak_gib: Optional[float] = None
    loss: Any = None
    err: str = ""
    applied: str = ""
    beat_naive: bool = False
    beat_eager: bool = False
    reason: str = ""
    vs_naive_pct: Optional[float] = None
    residual: dict = field(default_factory=dict)
    correctness_ok: Optional[bool] = None
    correctness: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoundRecord:
    idx: int
    mode: str  # "optimize" | "repair"
    recipe: dict = field(default_factory=dict)
    eval: Optional[EvalResult] = None
    llm_raw: str = ""

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "mode": self.mode,
            "recipe": self.recipe,
            "eval": self.eval.to_dict() if self.eval else None,
            "llm_raw": self.llm_raw[:4000],
        }
