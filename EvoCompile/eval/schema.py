"""Unified GKO eval CSV schema for Suite A (Dynamo) and Suite B (TorchTitan)."""

from __future__ import annotations

FIELDS = [
    "case_id",
    "suite",
    "model",
    "task",  # training | inference
    "gpu_count",
    "parallelism",
    "compile_config",
    "eager_or_uncompiled_step_ms",
    "vanilla_compile_step_ms",
    "agent_step_ms",
    "compile_time",
    "peak_memory",
    "graph_breaks",
    "recompiles",
    "correctness",
    "status",
]

EMPTY = ""


def empty_row(**kwargs) -> dict:
    row = {k: EMPTY for k in FIELDS}
    row.update(kwargs)
    return row
