"""RewriteWriter: scheme-conditioned Python for graph-break repairs.

TritonWriter emits kernels under eval/kernels/generated/.
This emits in-memory Module patches under eval/rewrites/generated/.

Recipe still JSON: {"op":"apply_rewrite","rewrite":"<id>"}.
The .py is the new scheme's code — not a lookup in recipe.py if/elif.

Generate IN: Observation (break_sample, break_kinds, forward_src sketch).
Generate OUT: rewrite id + source file.
Repair IN: current source + traceback from apply/compile.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from agent import GKO
from agent.break_kinds import classify_breaks
from agent.llm import LLMClient, parse_json_object
from agent.state import Observation

GENERATED_DIR = GKO / "eval" / "rewrites" / "generated"
CATALOG_DIR = GKO / "eval" / "rewrites"
_PLACEHOLDER_REWRITES = frozenset({"", "__generate__", "generate"})

_BANNED = ("subprocess", "os.system", "shutil.rmtree", "eval(", "exec(", "__import__")
_REWRITE_MARKERS = (
    "rewrites.generated",
    "apply_rewrite",
    "gko rewrite",
    "eval/rewrites/generated",
)


def is_catalog_rewrite(name: str) -> bool:
    n = str(name or "")
    if n in _PLACEHOLDER_REWRITES:
        return False
    return (CATALOG_DIR / f"{n}.py").is_file()


def recipe_wants_generated_rewrite(recipe: dict | None, scheme_name: str = "") -> bool:
    for a in (recipe or {}).get("actions") or []:
        if a.get("op") != "apply_rewrite":
            continue
        rid = a.get("rewrite") or ""
        if rid in _PLACEHOLDER_REWRITES:
            return True
    return False


def inject_rewrite_id(recipe: dict, rewrite_id: str) -> dict:
    """Bind a generated rewrite id. Never overwrite a catalog hook (fused_sdpa_attn)."""
    out = dict(recipe)
    acts = []
    found = False
    for a in list(out.get("actions") or []):
        a = dict(a)
        if a.get("op") == "apply_rewrite":
            rid = str(a.get("rewrite") or "")
            if rid in _PLACEHOLDER_REWRITES or rid == rewrite_id:
                a["rewrite"] = rewrite_id
                found = True
            elif is_catalog_rewrite(rid):
                acts.append(a)
                continue
            else:
                a["rewrite"] = rewrite_id
                found = True
        acts.append(a)
    if not found:
        if any(
            a.get("op") == "apply_rewrite" and is_catalog_rewrite(str(a.get("rewrite") or ""))
            for a in acts
        ):
            out["actions"] = acts
            return out
        acts = [{"op": "apply_rewrite", "rewrite": rewrite_id}] + acts
    out["actions"] = acts
    return out


def is_rewrite_error(err: str, recipe: dict | None = None) -> bool:
    """True only for generated S1 rewrites. Catalog hooks are not RewriteWriter jobs."""
    e = err or ""
    acts = (recipe or {}).get("actions") or []
    rids = [str(a.get("rewrite") or "") for a in acts if a.get("op") == "apply_rewrite"]
    if rids and all(is_catalog_rewrite(r) for r in rids):
        return False
    if any(m in e for m in _REWRITE_MARKERS):
        return True
    if any(a.get("op") == "apply_rewrite" for a in acts) and "Traceback" in e:
        if any(r in _PLACEHOLDER_REWRITES or r.startswith("g_") for r in rids):
            if "triton" not in e.lower() and "gko::" not in e:
                return True
    return False


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", (text or "rewrite").lower()).strip("_")
    return (s or "rewrite")[:40]


def _extract_source(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", t)
    if m:
        return m.group(1).strip() + "\n"
    if t.startswith("{"):
        obj = parse_json_object(t)
        if obj and isinstance(obj.get("source"), str):
            return obj["source"]
        if obj and isinstance(obj.get("code"), str):
            return obj["code"]
    return t if "def apply" in t else ""


def _validate_source(src: str) -> str | None:
    if not src or "def apply" not in src:
        return "missing def apply(model)"
    for b in _BANNED:
        if b in src:
            return f"banned token {b}"
    try:
        ast.parse(src)
    except SyntaxError as e:
        return f"syntax: {e}"
    return None


def write_rewrite(rewrite_id: str, source: str) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    init = GENERATED_DIR / "__init__.py"
    if not init.exists():
        init.write_text("")
    path = GENERATED_DIR / f"{rewrite_id}.py"
    path.write_text(source if source.endswith("\n") else source + "\n")
    return path


def heuristic_source(obs: Observation) -> str:
    """No-LLM codegen: still a real .py, specialized from break taxonomy."""
    tagged = classify_breaks(obs.features.get("break_sample") or [])
    kinds = tagged.get("break_kinds") or obs.features.get("break_kinds") or []
    sample = [str(x)[:160] for x in (obs.features.get("break_sample") or [])[:8]]
    side = bool(tagged.get("has_side_effect")) or "side_effect" in kinds
    data = bool(tagged.get("has_item") or tagged.get("has_nonzero")) or "data_dependent" in kinds
    setattr_b = bool(tagged.get("has_setattr")) or "setattr" in kinds
    lines = [
        '"""GKO generated rewrite (heuristic). Replace via RewriteWriter+LLM."""',
        "from __future__ import annotations",
        "",
        "BREAK_SAMPLE = " + repr(sample),
        "BREAK_KINDS = " + repr(list(kinds)),
        "",
        "def apply(model):",
        "    import torch",
        "    notes = []",
    ]
    if side:
        lines += [
            "    bag = getattr(torch._dynamo.config, 'reorderable_logging_functions', None)",
            "    if bag is not None and hasattr(bag, 'add'):",
            "        bag.add(print)",
            "        notes.append('reorderable_print')",
        ]
    if data:
        lines += [
            "    if hasattr(torch._dynamo.config, 'capture_scalar_outputs'):",
            "        torch._dynamo.config.capture_scalar_outputs = True",
            "        notes.append('capture_scalar_outputs')",
        ]
    if setattr_b:
        lines += [
            "    # setattr / store_attr: leave matching leaves eager (no disk edit).",
            "    from torch._dynamo import disable as _dyn_disable",
            "    n = 0",
            "    blob = ' '.join(BREAK_SAMPLE).lower()",
            "    for mod in model.modules():",
            "        name = type(mod).__name__",
            "        if name.lower() in blob or any(name in s for s in BREAK_SAMPLE):",
            "            if not getattr(mod.forward, '__dynamo_disable', False):",
            "                mod.forward = _dyn_disable(mod.forward)",
            "                n += 1",
            "    notes.append(f'disable_forward n={n}')",
        ]
    lines += [
        "    return ','.join(notes) or 'noop'",
        "",
    ]
    return "\n".join(lines)


class RewriteWriterAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    def generate(
        self,
        obs: Observation,
        *,
        recipe: dict | None = None,
        case_tag: str = "",
    ) -> tuple[str, str, str]:
        kid = _slug(case_tag or f"g_{obs.model}_s1")
        if self.llm is None:
            src = heuristic_source(obs)
            write_rewrite(kid, src)
            return kid, src, "heuristic: RewriteWriter wrote generated module (no LLM)"
        from agent.prompts import SYSTEM_REWRITE, user_rewrite_generate

        tagged = classify_breaks(obs.features.get("break_sample") or [])
        user = user_rewrite_generate(
            case=obs.model,
            rewrite_id_hint=kid,
            break_sample=list(obs.features.get("break_sample") or [])[:8],
            break_kinds=tagged.get("break_kinds") or [],
            forward_src=(obs.forward_src or "")[:2500],
            model_tree=(obs.model_tree or "")[:1500],
        )
        raw = self.llm.chat(SYSTEM_REWRITE, user)
        src = _extract_source(raw)
        err = _validate_source(src) if src else "no python in reply"
        if err:
            src = heuristic_source(obs)
            write_rewrite(kid, src)
            return kid, src, f"generate fallback heuristic: {err}\n{(raw or '')[:1500]}"
        write_rewrite(kid, src)
        return kid, src, raw

    def repair(
        self,
        obs: Observation,
        *,
        rewrite_id: str,
        source: str,
        error: str,
        history: list | None = None,
    ) -> tuple[str, str, str]:
        path = GENERATED_DIR / f"{rewrite_id}.py"
        cur = source or (path.read_text() if path.exists() else "")
        if self.llm is None:
            src = heuristic_source(obs)
            write_rewrite(rewrite_id, src)
            return rewrite_id, src, "heuristic rewrite repair"
        from agent.prompts import SYSTEM_REWRITE, user_rewrite_repair

        raw = self.llm.chat(
            SYSTEM_REWRITE,
            user_rewrite_repair(
                case=obs.model,
                rewrite_id=rewrite_id,
                source=cur[-12000:],
                error=error[-6000:],
                history=history,
            ),
        )
        src = _extract_source(raw)
        err = _validate_source(src) if src else "no python in reply"
        if err:
            src = heuristic_source(obs)
            write_rewrite(rewrite_id, src)
            return rewrite_id, src, f"repair fallback heuristic: {err}\n{(raw or '')[:1500]}"
        write_rewrite(rewrite_id, src)
        return rewrite_id, src, raw
