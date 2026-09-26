"""Retrieve a compile scheme from the memory/treememory decision tree."""

from __future__ import annotations

from pathlib import Path

from agent.casememory import attach_casememory
from agent.memory_tree import MemoryStore, TREE_PATH
from agent.state import Observation, SchemeHit


class RetrieverAgent:
    """Deterministic hard match. LLM does not choose the scheme name."""

    def __init__(self, tree_path: Path | None = None):
        self.tree_path = tree_path or TREE_PATH
        self.store = MemoryStore.load(self.tree_path)

    def reload(self) -> None:
        self.store.reload()

    def retrieve(self, obs: Observation) -> SchemeHit:
        hit, _path = self.store.walk(obs)
        attach_casememory(hit)
        return hit
