"""Load / walk / mutate the retrieval tree in memory/treememory.

The file is JSON. Hard match is first-child-wins on ordered `when` predicates.
Evolve inserts sibling exception nodes; similarity never dumps this whole file
into an LLM — it only reads the compact `cases` cards.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agent.break_kinds import attach_mark_dynamic
from agent.fingerprint import compact_fp, context_from_fp, match_context
from agent.state import Observation, SchemeHit

GKO = Path(__file__).resolve().parents[1]
TREE_PATH = GKO / "memory" / "treememory"

ACCEL_SCHEMES = {
    "S1",
    "S2",
    "S3",
    "S3-fc",
    "S3-step",
    "S3-embed",
    "S3-deep",
    "S3-unfused",
    "S3b",
    "S3-tiny-vit",
    "S5+S3",
    "S6-fuse",
    "S6-sdpa",
}

# Failure-derived stop leaves: do not "explore away" from them into S3*.
STOP_SCHEMES = {
    "B-loader",
    "S4",
    "S5-qat",
    "S5-opacus",
    "S5-embed",
    "F6",
    "F2-eager",
    "F2-skinny",
}

# After a scheme is tried, which named schemes are legal next probes.
DEFAULT_EXPLORE_FROM = {
    "S3-unfused": ["S6-sdpa", "S6-fuse", "S3b", "S3-step", "identity"],
    "S3b": ["S6-sdpa", "S6-fuse", "S3-step", "S3-deep", "identity"],
    "S3-deep": ["S6-sdpa", "S3-step", "S3b", "identity"],
    "S3-embed": ["S3b", "S3-step", "identity"],
    "S3-tiny-vit": ["S3b", "S3-step", "identity"],
    "S3-fc": ["S3-step", "identity", "F2-cnn"],
    "S3-step": ["S6-sdpa", "S6-fuse", "S3b", "S3-deep", "identity"],
    "S2": ["S6-sdpa", "S3b", "S3-step", "identity"],
    "S1": ["S3b", "identity"],
    "S5+S3": ["S3-step", "S5", "identity"],
    "S5": ["S5+S3", "identity"],
    "F2-cnn": ["S6-fuse", "S3-step", "S3-fc"],
    "F2-skinny": ["S6-sdpa"],
    "F6": ["S3-step"],
    "F2-eager": [],
    "identity": ["S6-sdpa", "S6-fuse", "S3-step", "S3b", "S3-fc"],
    "S6-fuse": ["S3-step", "S3b", "identity"],
    "S6-sdpa": ["S6-fuse", "identity"],
    "S4": [],
    "S5-qat": [],
    "S5-opacus": [],
    "S5-embed": [],
    "B-loader": [],
}

BAN_DIST = 0.28
WIN_PCT = 5.0
LOSE_PCT = -5.0
PATH_DIST = 0.32
TRIAL_CAP = 2000
PATH_CAP = 200
VERDICT_RANK = {"win": 3, "tie": 2, "lose": 1, "crash": 0}

_SUFFIX_OPS = (
    ("_gte", lambda a, b: a is not None and a >= b),
    ("_gt", lambda a, b: a is not None and a > b),
    ("_lte", lambda a, b: a is not None and a <= b),
    ("_lt", lambda a, b: a is not None and a < b),
    ("_eq", lambda a, b: a == b),
)


def _num(ctx: dict, key: str):
    v = ctx.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def eval_when(when: Any, ctx: dict) -> bool:
    if not when:
        return True
    if not isinstance(when, dict):
        return False
    parts: list[bool] = []
    if "and" in when:
        parts.append(all(eval_when(x, ctx) for x in when["and"]))
    if "or" in when:
        parts.append(any(eval_when(x, ctx) for x in when["or"]))
    if "not" in when:
        parts.append(not eval_when(when["not"], ctx))
    for k, v in when.items():
        if k in ("and", "or", "not"):
            continue
        parts.append(_atom(k, v, ctx))
    return all(parts) if parts else True


def _atom(k: str, v: Any, ctx: dict) -> bool:
    if k == "name_contains":
        name = str(ctx.get("name") or "").lower()
        items = v if isinstance(v, list) else [v]
        return any(str(x).lower() in name for x in items)
    if k == "opt_contains":
        opt = str(ctx.get("opt") or "")
        items = v if isinstance(v, list) else [v]
        return any(str(x) in opt for x in items)
    if k == "breaks_contains":
        s = str(ctx.get("breaks_s") or "")
        items = v if isinstance(v, list) else [v]
        return any(str(x) in s for x in items)
    if k == "model_type_contains":
        mt = str(ctx.get("model_type") or "").lower()
        items = v if isinstance(v, list) else [v]
        return any(str(x).lower() in mt for x in items)
    if k == "arch_eq":
        return str(ctx.get("arch_class") or "") == str(v)

    base = k
    op = None
    for sfx, fn in _SUFFIX_OPS:
        if k.endswith(sfx) and len(k) > len(sfx):
            base = k[: -len(sfx)]
            op = fn
            break

    key_aliases = {
        "conv": "conv",
        "lin": "lin",
        "n_params": "n_params",
        "vanilla_ms": "vanilla_ms",
        "eager_ms": "eager_ms",
        "ratio": "ratio",
        "fwd_breaks": "fwd_breaks",
        "n_layer": "n_layer",
        "hidden": "hidden",
        "vocab": "vocab",
        "n_tensors": "n_tensors",
        "sync": "sync",
        "layerdrop_n": "layerdrop_n",
        "dense_n": "dense_n",
        "vocab_size": "vocab",
        "n_params_m": "n_params",
        "conv_param_frac": "conv",
        "linear_param_frac": "lin",
    }
    ck = key_aliases.get(base, base)

    if op is not None:
        return op(_num(ctx, ck), float(v))

    cv = ctx.get(ck)
    if v is False:
        return cv is False
    if v is True:
        return cv is True
    if v is None:
        return cv is None
    return cv == v


def identity_recipe(model: str, note: str, **extra) -> dict:
    r = {
        "model": model,
        "backend": "inductor",
        "fullgraph": False,
        "note": note,
        "actions": [{"op": "compile", "path": ""}],
    }
    r.update(extra)
    return r


def compile_adam_recipe(model: str, note: str, fullgraph: bool) -> dict:
    return {
        "model": model,
        "backend": "inductor",
        "fullgraph": False,
        "compile_optimizer": True,
        "note": note,
        "actions": [{"op": "compile", "path": "", "fullgraph": fullgraph, "dynamic": False}],
    }


def materialize_recipe(model: str, spec: dict | None, scheme: str) -> dict:
    spec = spec or {}
    kind = spec.get("kind") or "identity"
    note = spec.get("note") or scheme
    fg = spec.get("fullgraph")
    prefix = spec.get("fullgraph_if_prefix")
    if prefix:
        fg = str(model).lower().startswith(str(prefix).lower())
    if fg is None:
        fg = False

    if kind == "skip_compile":
        return {
            "model": model,
            "backend": "inductor",
            "fullgraph": False,
            "note": note,
            "actions": [],
        }
    if kind == "compile_adam":
        return compile_adam_recipe(model, note, bool(fg))
    if kind == "s4_detectron":
        return {
            "model": model,
            "backend": "inductor",
            "fullgraph": False,
            "inductor": {"cudagraphs": False},
            "note": note,
            "actions": [
                {"op": "disable_forward", "path": ""},
                {"op": "compile", "path": "backbone", "fullgraph": False, "dynamic": False},
            ],
        }
    if kind == "s5_opacus":
        return {
            "model": model,
            "backend": "inductor",
            "fullgraph": False,
            "note": note,
            "actions": [
                {"op": "disable_forward", "path": ""},
                {"op": "compile", "path": "_module"},
                {"op": "disable_method", "path": "", "name": "capture_activations_hook"},
                {"op": "disable_method", "path": "", "name": "capture_backprops_hook"},
            ],
        }
    if kind == "s1_yolo":
        return {
            "model": model,
            "backend": "inductor",
            "fullgraph": False,
            "note": note,
            "actions": [
                {"op": "rewrite_class_forward", "class": "YOLOLayer", "kind": "yolo_layer_train_no_grid"},
                {"op": "compile", "path": "", "fullgraph": True, "dynamic": False},
            ],
        }
    if kind == "s1_speech":
        return {
            "model": model,
            "backend": "inductor",
            "fullgraph": False,
            "dynamo": {"capture_scalar_outputs": True},
            "note": note,
            "actions": [
                {"op": "speech_vectorize_pad_mask"},
                {"op": "compile", "path": "", "fullgraph": True, "dynamic": False},
            ],
        }
    if kind == "s1_logging":
        return {
            "model": model,
            "backend": "inductor",
            "fullgraph": False,
            "note": note,
            "actions": [
                {"op": "allow_logging"},
                {"op": "compile", "path": "", "fullgraph": True, "dynamic": False},
            ],
        }
    if kind == "s1_generic":
        return {
            "model": model,
            "backend": "inductor",
            "fullgraph": False,
            "note": note,
            "actions": [
                {"op": "apply_rewrite", "rewrite": "__generate__"},
                {"op": "compile", "path": "", "fullgraph": True, "dynamic": False},
            ],
        }
    if kind == "s2":
        return {
            "model": model,
            "backend": "inductor",
            "fullgraph": False,
            "compile_optimizer": True,
            "note": note,
            "actions": [
                {"op": "hf_layerdrop_off"},
                {"op": "compile", "path": "", "fullgraph": True, "dynamic": False},
            ],
        }
    if kind == "s3_step":
        return {
            "model": model,
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": note,
            "actions": [],
        }
    if kind == "s6_fuse":
        return {
            "model": model,
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": note,
            "actions": [
                {"op": "replace_pattern", "kernel": "__generate__"},
            ],
        }
    if kind == "s6_sdpa":
        return {
            "model": model,
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": note,
            "actions": s6_sdpa_actions(model),
        }
    extra = {k: v for k, v in spec.items() if k not in ("kind", "note", "fullgraph", "fullgraph_if_prefix")}
    return identity_recipe(model, note, **extra)


def s6_sdpa_actions(model: str) -> list:
    """DistilBert/OPT: in-tree F.sdpa. Others: generate a case kernel + hook attention."""
    n = (model or "").lower().replace("-", "_")
    if "distil" in n or n == "optforcausallm":
        return [{"op": "apply_rewrite", "rewrite": "sdpa_rewrite"}]
    return [
        {"op": "replace_pattern", "kernel": "__generate__"},
        {"op": "apply_rewrite", "rewrite": "fused_sdpa_attn"},
    ]


_S6_CATALOG_KERNELS = frozenset({"fused_sdpa", "addmm_gelu", "silu_mul"})


def _force_generate_kernel(acts: list) -> list:
    """Catalog ids are Writer format samples, never the S6 recipe kernel."""
    out = []
    seen = False
    for raw in acts:
        a = dict(raw)
        if a.get("op") != "replace_pattern":
            out.append(a)
            continue
        if seen:
            continue
        seen = True
        k = str(a.get("kernel") or "")
        if k.lower() in _S6_CATALOG_KERNELS or k in ("", "generate"):
            k = "__generate__"
        out.append({"op": "replace_pattern", "kernel": k or "__generate__"})
    return out


def normalize_s6_recipe(recipe: dict | None, scheme_name: str, model: str = "") -> dict:
    """Force the insert contract: kernel/rewrite before compile(train_step), never nest compile(model)."""
    out = dict(recipe or {})
    name = scheme_name or ""
    if name not in ("S6-fuse", "S6-sdpa"):
        return out
    model = model or str(out.get("model") or "")
    out["compile_step"] = True
    out["compile_optimizer"] = False
    acts = [dict(a) for a in (out.get("actions") or []) if str(a.get("op") or "") != "compile"]
    acts = _force_generate_kernel(acts)
    spec = out.get("kernel_spec")
    if isinstance(spec, dict):
        spec = dict(spec)
        spec["catalog_hint"] = None
        out["kernel_spec"] = spec
    if name == "S6-fuse":
        if not any(a.get("op") == "replace_pattern" for a in acts):
            acts.insert(0, {"op": "replace_pattern", "kernel": "__generate__"})
    else:
        desired = s6_sdpa_actions(model)
        rewrites = [str(a.get("rewrite") or "") for a in acts if a.get("op") == "apply_rewrite"]
        n = model.lower().replace("-", "_")
        distil_or_opt = "distil" in n or n == "optforcausallm"
        catalog_hooks = {"fused_sdpa_attn", "sdpa_rewrite"}
        generated_rw = [
            r for r in rewrites if r and r not in catalog_hooks and r not in ("__generate__", "generate")
        ]
        if generated_rw:
            # Repair must not swap the attention hook for an S1 rewrite (g_s1).
            acts = [a for a in acts if not (
                a.get("op") == "apply_rewrite" and str(a.get("rewrite") or "") in generated_rw
            )]
            rewrites = [str(a.get("rewrite") or "") for a in acts if a.get("op") == "apply_rewrite"]
        if not any(a.get("op") == "apply_rewrite" for a in acts):
            acts = desired
        elif not distil_or_opt and rewrites == ["sdpa_rewrite"]:
            acts = desired
        if not distil_or_opt and not any(a.get("op") == "replace_pattern" for a in acts):
            acts.insert(0, {"op": "replace_pattern", "kernel": "__generate__"})
        if distil_or_opt:
            acts = [a for a in acts if a.get("op") != "replace_pattern"]
    out["actions"] = acts
    out.setdefault("model", model)
    out.setdefault("backend", "inductor")
    out.setdefault("fullgraph", False)
    return out


def verdict_of(*, ok, beat_naive, vs_naive_pct) -> str:
    if not ok:
        return "crash"
    vs = vs_naive_pct
    if beat_naive or (vs is not None and float(vs) >= WIN_PCT):
        return "win"
    if vs is not None and float(vs) <= LOSE_PCT:
        return "lose"
    return "tie"


def trial_better(new: dict, old: dict) -> bool:
    nr = VERDICT_RANK.get(new.get("verdict") or "", 0)
    or_ = VERDICT_RANK.get(old.get("verdict") or "", 0)
    if nr != or_:
        return nr > or_
    nv = new.get("vs_naive_pct")
    ov = old.get("vs_naive_pct")
    nv = float(nv) if nv is not None else -999.0
    ov = float(ov) if ov is not None else -999.0
    return nv > ov


def trial_from_case(case: dict) -> dict:
    t = dict(case)
    ok = t.get("ok")
    if ok is None:
        ok = t.get("agent_ms") is not None
    t["ok"] = bool(ok)
    t["verdict"] = verdict_of(
        ok=t["ok"],
        beat_naive=t.get("beat_naive"),
        vs_naive_pct=t.get("vs_naive_pct"),
    )
    return t


def _knn(fp: dict, entries: list[dict], *, k: int, exclude_case: str = "") -> list[tuple[float, dict]]:
    from agent.fingerprint import fingerprint_distance

    scored: list[tuple[float, dict]] = []
    for entry in entries:
        if exclude_case and entry.get("case") == exclude_case:
            continue
        other = entry.get("fp") or {}
        scored.append((fingerprint_distance(fp, other), entry))
    scored.sort(key=lambda x: x[0])
    return scored[:k]


S1_LOGGING_LEAF = {
    "id": "S1-logging",
    "kind": "leaf",
    "when": {"has_side_effect": True},
    "scheme": "S1",
    "reason": "print/logging/IO graph-break; allow_logging then fullgraph",
    "recipe": {"kind": "s1_logging", "note": "S1: reorder print/logging then fullgraph"},
    "do_not": ["AST-edit user forward to delete print", "Titan per-Bottleneck compile (F1)"],
    "examples": "",
    "confidence": "hard",
}

S6_SDPA_LEFTOVER_LEAF = {
    "id": "S6-sdpa",
    "kind": "leaf",
    "when": {"unfused_bmm_attention": True},
    "scheme": "S6-sdpa",
    "reason": "unfused bmm+softmax leftover (no flash ATen); hook attention + compile(train_step)",
    "recipe": {"kind": "s6_sdpa", "note": "S6-sdpa fused attention rewrite + compile_step"},
    "do_not": [
        "custom Triton fused_sdpa first on DistilBert",
        "compiled Adam on DistilBert",
        "nest compile(model)+compile_step",
        "handwritten tiled CUTLASS FA",
        "drop kernel/rewrite to pass rel_loss",
    ],
    "examples": "OPT vendor FMHA 1.28×; Longformer 1.37×; DistilBert explore after F2-skinny",
    "confidence": "hard",
}

S6_FUSE_ATTN_LEAF = {
    "id": "S6-fuse-attn",
    "kind": "leaf",
    "when": {"has_unfused_epilogue": True},
    "scheme": "S6-fuse",
    "reason": (
        "Linear+GELU / leftover poi after official wrap; "
        "CUTLASS tensor-op GEMM+act (addmm_gelu / SwiGLU epi), never Triton tl.dot GEMM; "
        "Triton only for pure pointwise leftovers; then compile_step"
    ),
    "recipe": {"kind": "s6_fuse", "note": "S6 catalog/generate fused kernel then compile_step"},
    "do_not": [
        "naive tl.dot vs cuBLAS",
        "primary-path Triton tl.dot GEMM",
        "handwritten float / TC-less CUDA GEMM",
        "opaque silu_and_mul as fake epilogue",
        "nest compile(model)+compile_step",
        "drop kernel to pass rel_loss",
    ],
    "examples": "CUTLASS GEMM+GELU on leftover addmm+gelu; CUTLASS GEMM+SwiGLU for FFN",
    "confidence": "hard",
}

S6_FUSE_CNN_LEAF = {
    "id": "S6-fuse-cnn",
    "kind": "leaf",
    "when": {"has_unfused_epilogue": True},
    "scheme": "S6-fuse",
    "reason": "CNN leftover after official compile(train_step): silu_mul / launch-HBM poi; catalog kernel then compile_step",
    "recipe": {"kind": "s6_fuse", "note": "S6 catalog/generate fused kernel then compile_step"},
    "do_not": [
        "naive tl.dot vs cuBLAS",
        "CUTLASS conv vs cuDNN",
        "nest compile(model)+compile_step",
        "drop kernel to pass rel_loss",
    ],
    "examples": "silu_mul: mobilenetv2_100 1.07× vs official step",
    "confidence": "hard",
}


def _insert_after(children: list, after_id: str, new_node: dict) -> bool:
    if any(c.get("id") == new_node.get("id") for c in children):
        return False
    idx = next((i for i, c in enumerate(children) if c.get("id") == after_id), None)
    if idx is None:
        children.append(new_node)
    else:
        children.insert(idx + 1, new_node)
    return True


def _insert_before(children: list, before_id: str, new_node: dict) -> bool:
    if any(c.get("id") == new_node.get("id") for c in children):
        return False
    idx = next((i for i, c in enumerate(children) if c.get("id") == before_id), None)
    if idx is None:
        children.append(new_node)
    else:
        children.insert(idx, new_node)
    return True


def _relax_s6_sdpa_name_filter(node: dict) -> bool:
    if node.get("id") != "S6-sdpa" or node.get("kind") != "leaf":
        return False
    when = node.get("when") or {}
    if when == {"unfused_bmm_attention": True}:
        return False
    node["when"] = {"unfused_bmm_attention": True}
    node["reason"] = (
        "unfused bmm+softmax leftover (no flash ATen); hook attention + compile(train_step)"
    )
    return True


def _ensure_s6_feature_leaves(node: dict) -> bool:
    """S6 retrieve by leftover I-channel, not DistilBert/OPT name whitelist."""
    if not isinstance(node, dict):
        return False
    changed = False
    nid = node.get("id") or ""
    kids = list(node.get("children") or [])
    if nid == "adam-zero-break":
        if _insert_after(kids, "F2-skinny", dict(S6_SDPA_LEFTOVER_LEAF)):
            changed = True
        if _insert_after(kids, "S6-sdpa", dict(S6_FUSE_ATTN_LEAF)):
            changed = True
        if changed:
            node["children"] = kids
    elif nid == "cnn-0break":
        if _insert_before(kids, "F2-cnn", dict(S6_FUSE_CNN_LEAF)):
            node["children"] = kids
            changed = True
    if _relax_s6_sdpa_name_filter(node):
        changed = True
    for child in node.get("children") or []:
        if _ensure_s6_feature_leaves(child):
            changed = True
    return changed


def _ensure_s1_logging_leaf(node: dict) -> bool:
    """Insert S1-logging under S1-hot (before S1-generic) without rewriting evolve siblings."""
    if not isinstance(node, dict):
        return False
    changed = False
    if node.get("id") == "S1-hot":
        children = list(node.get("children") or [])
        if not any(c.get("id") == "S1-logging" for c in children):
            idx = next(
                (i for i, c in enumerate(children) if c.get("id") == "S1-generic"),
                len(children),
            )
            children.insert(idx, dict(S1_LOGGING_LEAF))
            node["children"] = children
            changed = True
    for child in node.get("children") or []:
        if _ensure_s1_logging_leaf(child):
            changed = True
    return changed


class MemoryStore:
    def __init__(self, data: dict, path: Path = TREE_PATH):
        self.data = data
        self.path = path

    @classmethod
    def load(cls, path: Path | None = None) -> "MemoryStore":
        path = path or TREE_PATH
        if not path.exists() or path.stat().st_size < 8:
            from agent.memory_bootstrap import write_default

            write_default(path)
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict) or "tree" not in data:
            raise ValueError(f"treememory missing tree: {path}")
        store = cls(data, path)
        if store.ensure_schema():
            try:
                store.save(backup=False)
            except OSError:
                pass
        return store

    def reload(self) -> None:
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, *, backup: bool = False) -> None:
        if backup and self.path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = self.path.parent / f"treememory.bak.{ts}"
            shutil.copy2(self.path, bak)
            old = sorted(self.path.parent.glob("treememory.bak.*"))
            for p in old[:-5]:
                p.unlink(missing_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def schemes(self) -> dict:
        return self.data.get("schemes") or {}

    def cases(self) -> list[dict]:
        return list(self.data.get("cases") or [])

    def trials(self) -> list[dict]:
        return list(self.data.get("trials") or [])

    def explore_from(self) -> dict:
        return self.data.get("explore_from") or dict(DEFAULT_EXPLORE_FROM)

    def paths(self) -> list[dict]:
        return list(self.data.get("paths") or [])

    def ensure_schema(self) -> bool:
        """Fill trials / explore_from / paths on older treememory files. True if mutated."""
        changed = False
        if "explore_from" not in self.data:
            self.data["explore_from"] = dict(DEFAULT_EXPLORE_FROM)
            changed = True
        if "paths" not in self.data:
            self.data["paths"] = []
            changed = True
        if not self.data.get("trials"):
            self.data["trials"] = [trial_from_case(c) for c in self.cases()]
            changed = True
        if _ensure_s1_logging_leaf(self.data.get("tree") or {}):
            changed = True
        if _ensure_s6_feature_leaves(self.data.get("tree") or {}):
            changed = True
        return changed

    def scheme_one_liners(self, names: list[str] | None = None) -> dict[str, str]:
        out = {}
        cat = self.schemes()
        for n, spec in cat.items():
            if names is not None and n not in names:
                continue
            out[n] = spec.get("one_line") or n
        return out

    def recipe_for_scheme(self, scheme: str, model: str) -> dict:
        spec = (self.schemes().get(scheme) or {}).get("recipe") or {"kind": "identity", "note": scheme}
        return materialize_recipe(model, spec, scheme)

    def walk(self, obs: Observation) -> tuple[SchemeHit, list[str]]:
        ctx = match_context(obs)
        return self.walk_ctx(ctx, model=obs.model)

    def walk_ctx(self, ctx: dict, *, model: str = "") -> tuple[SchemeHit, list[str]]:
        model = model or str(ctx.get("name") or "")
        path: list[str] = []
        node = self.data.get("tree") or {}
        leaf = self._first_leaf(node, ctx, path)
        if leaf is None:
            hit = SchemeHit(
                name="identity",
                reason="no scheme matched; identity = naive compile(model)",
                recipe_hint=identity_recipe(model, "identity fallback"),
                source="hard",
                confidence="low",
                path=path,
            )
            return hit, path
        scheme = leaf.get("scheme") or "identity"
        reason = leaf.get("reason") or scheme
        recipe = materialize_recipe(model, leaf.get("recipe"), scheme)
        recipe = attach_mark_dynamic(recipe, ctx)
        hit = SchemeHit(
            name=scheme,
            reason=reason,
            recipe_hint=recipe,
            do_not=list(leaf.get("do_not") or []),
            examples=leaf.get("examples") or "",
            source="hard",
            confidence=leaf.get("confidence") or "hard",
            path=path + [leaf.get("id") or scheme],
            node_id=leaf.get("id") or "",
        )
        return hit, hit.path

    def _first_leaf(self, node: dict, ctx: dict, path: list[str]) -> Optional[dict]:
        if not node:
            return None
        nid = node.get("id") or ""
        if nid:
            path.append(nid)
        kind = node.get("kind") or ("leaf" if node.get("scheme") else "switch")
        if kind == "leaf":
            return node
        for child in node.get("children") or []:
            if not eval_when(child.get("when"), ctx):
                continue
            child_kind = child.get("kind") or ("leaf" if child.get("scheme") else "switch")
            if child_kind == "leaf":
                return child
            mark = len(path)
            found = self._first_leaf(child, ctx, path)
            if found is not None:
                return found
            del path[mark:]
        return None

    def find_node(self, node_id: str, node: dict | None = None) -> Optional[dict]:
        node = node if node is not None else (self.data.get("tree") or {})
        if node.get("id") == node_id:
            return node
        for ch in node.get("children") or []:
            found = self.find_node(node_id, ch)
            if found is not None:
                return found
        return None

    def parent_of(self, node_id: str, node: dict | None = None) -> Optional[dict]:
        node = node if node is not None else (self.data.get("tree") or {})
        for ch in node.get("children") or []:
            if ch.get("id") == node_id:
                return node
            found = self.parent_of(node_id, ch)
            if found is not None:
                return found
        return None

    def insert_before(self, sibling_id: str, new_node: dict) -> bool:
        parent = self.parent_of(sibling_id)
        if parent is None:
            return False
        kids = list(parent.get("children") or [])
        idx = next((i for i, c in enumerate(kids) if c.get("id") == sibling_id), None)
        if idx is None:
            return False
        kids.insert(idx, new_node)
        parent["children"] = kids
        return True

    def record_trial(self, entry: dict, *, write: bool = True) -> dict:
        """Append-only win/lose/crash log. Never drops a previous attempt."""
        trial = trial_from_case(entry)
        trials = list(self.data.get("trials") or [])
        trials.append(trial)
        self.data["trials"] = trials[-TRIAL_CAP:]
        self._upsert_best_case(trial)
        self.data["updated"] = datetime.now().isoformat(timespec="seconds")
        if write:
            self.save(backup=False)
        return trial

    def record_case(self, entry: dict, *, write: bool = True) -> None:
        """Back-compat: treat as a trial (append) plus best-so-far update."""
        self.record_trial(entry, write=write)

    def _upsert_best_case(self, trial: dict) -> None:
        cases = list(self.data.get("cases") or [])
        key = trial.get("case")
        old = next((c for c in cases if c.get("case") == key), None)
        if old is None or trial_better(trial, old):
            cases = [c for c in cases if c.get("case") != key]
            cases.append(dict(trial))
            self.data["cases"] = cases

    def neighbors(
        self,
        fp: dict,
        *,
        polarity: str = "all",
        k: int = 5,
        exclude_case: str = "",
    ) -> list[tuple[float, dict]]:
        pool = self.trials() or self.cases()
        if polarity == "win":
            pool = [t for t in pool if t.get("verdict") == "win" or t.get("beat_naive")]
        elif polarity == "lose":
            pool = [t for t in pool if t.get("verdict") in ("lose", "crash")]
        elif polarity == "tie":
            pool = [t for t in pool if t.get("verdict") == "tie"]
        return _knn(fp, pool, k=k, exclude_case=exclude_case)

    def banned_schemes(
        self,
        fp: dict,
        *,
        tried: list[str] | None = None,
        exclude_case: str = "",
    ) -> list[str]:
        banned = list(tried or [])
        for dist, entry in self.neighbors(fp, polarity="lose", k=8, exclude_case=exclude_case):
            if dist > BAN_DIST:
                continue
            sch = entry.get("scheme")
            if sch and sch not in banned and sch in ACCEL_SCHEMES:
                banned.append(sch)
        path = self.nearest_path(fp, exclude_case=exclude_case)
        if path and path.get("dist", 1) <= PATH_DIST:
            best = path.get("best")
            for sch in path.get("tried") or []:
                if sch and sch != best and sch not in banned:
                    banned.append(sch)
        return banned

    def nearest_path(self, fp: dict, *, exclude_case: str = "") -> Optional[dict]:
        ranked = _knn(fp, self.paths(), k=1, exclude_case=exclude_case)
        if not ranked:
            return None
        dist, entry = ranked[0]
        out = dict(entry)
        out["dist"] = dist
        return out

    def record_path(
        self,
        *,
        case: str,
        fp: dict,
        tried: list[str],
        outcomes: list[dict],
        best: str,
        best_vs_naive_pct,
        write: bool = True,
    ) -> None:
        rec = {
            "case": case,
            "fp": fp,
            "tried": list(tried),
            "outcomes": outcomes[-12:],
            "best": best,
            "best_vs_naive_pct": best_vs_naive_pct,
        }
        paths = [p for p in self.paths() if p.get("case") != case]
        paths.append(rec)
        self.data["paths"] = paths[-PATH_CAP:]
        if write:
            self.save(backup=False)

    def log_evolve(self, event: dict, *, write: bool = True) -> None:
        log = list(self.data.get("evolve_log") or [])
        log.append(event)
        self.data["evolve_log"] = log[-80:]
        if write:
            self.save(backup=True)

    def hit_from_scheme(self, scheme: str, model: str, *, reason: str, source: str) -> SchemeHit:
        spec = self.schemes().get(scheme) or {}
        recipe = self.recipe_for_scheme(scheme, model)
        return SchemeHit(
            name=scheme,
            reason=reason,
            recipe_hint=recipe,
            do_not=list(spec.get("do_not") or []),
            examples=spec.get("examples") or "",
            source=source,
            confidence="medium" if source == "similarity" else "hard",
        )


def obs_case_entry(
    obs: Observation,
    *,
    scheme: str,
    result: dict | None = None,
    source: str = "run",
    recipe_kind: str = "",
) -> dict:
    fp = compact_fp(obs)
    entry = {
        "case": obs.model,
        "suite": obs.suite,
        "scheme": scheme,
        "fp": fp,
        "source": source,
        "recipe_kind": recipe_kind,
    }
    if result:
        entry["beat_naive"] = result.get("beat_naive")
        entry["vs_naive_pct"] = result.get("vs_naive_pct")
        entry["ok"] = result.get("ok")
        entry["agent_ms"] = result.get("ms")
        entry["verdict"] = verdict_of(
            ok=result.get("ok"),
            beat_naive=result.get("beat_naive"),
            vs_naive_pct=result.get("vs_naive_pct"),
        )
    return entry
