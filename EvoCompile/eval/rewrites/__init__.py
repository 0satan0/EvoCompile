"""Generated graph-break rewrites: Python modules with apply(model).

Unlike catalog kinds inlined in recipe.py, these files are written by
RewriteWriter (scheme → code). apply_recipe only loads the id.

Contract for eval/rewrites/generated/<id>.py:
  def apply(model) -> str
  Mutate the in-memory nn.Module (or Dynamo config). Do not write model source
  on disk. Idempotent. No eval/exec/subprocess.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REWRITES_DIR = Path(__file__).resolve().parent
GENERATED_DIR = REWRITES_DIR / "generated"


CATALOG_DIR = REWRITES_DIR


def list_rewrites() -> list[str]:
    names = []
    if GENERATED_DIR.is_dir():
        names.extend(
            p.stem
            for p in GENERATED_DIR.glob("*.py")
            if p.stem != "__init__"
        )
    names.extend(
        p.stem
        for p in CATALOG_DIR.glob("*.py")
        if p.stem not in ("__init__",)
    )
    return sorted(set(names))


def apply(model, name: str) -> str:
    if not name or name in ("__generate__", "generate"):
        raise ValueError(
            "apply_rewrite needs a concrete rewrite id; RewriteWriter must run first"
        )
    path = GENERATED_DIR / f"{name}.py"
    loc = f"rewrites.generated.{name}"
    if not path.is_file():
        path = CATALOG_DIR / f"{name}.py"
        loc = f"rewrites.{name}"
    if not path.is_file():
        raise ValueError(f"unknown rewrite {name!r}; generated={list_rewrites()}")
    spec = importlib.util.spec_from_file_location(loc, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load rewrite {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[loc] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "apply", None)
    if not callable(fn):
        raise ValueError(f"rewrite {name} missing def apply(model)")
    note = fn(model)
    return f"apply_rewrite:{name}" + (f" ({note})" if note else "")
