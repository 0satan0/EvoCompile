"""KernelWriter: LLM (or catalog) emits a kernel module that lands in-graph.

Called after Retriever/Optimizer chose S6-fuse, or a recipe already contains
replace_pattern with kernel=__generate__.

Known leftover kinds (Linear+GELU, silu*mul, dense MHA) have catalog
kernels as FORMAT examples. The Optimizer's kernel_spec decides whether
to copy a catalog hole or write a new fusion. Default is generate.
GEMM+act catalog is CUTLASS, not naive tl.dot.

Generate IN: fusion sites + insertion contract (naming, register(), FX pass).
Generate OUT: kernel id + source file under eval/kernels/generated/.
The recipe must replace_pattern BEFORE compile_step (compile(train_step));
never nest compile(model).
Repair IN: current source + Triton/CUTLASS traceback (not Dynamo/fullgraph).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from agent import GKO
from agent.fusion_sites import sites_from_features
from agent.llm import LLMClient, parse_json_object
from agent.state import Observation

GENERATED_DIR = GKO / "eval" / "kernels" / "generated"

_TRITON_MARKERS = (
    "triton.compiler",
    "triton.language",
    "CompilationError",
    "TTGIR",
    "ttgir",
    "gko::",
    "torch.ops.gko",
    "addmm_gelu",
    "kernels.generated",
    "kernels/generated",
    "eval/kernels",
    "custom_op",
    "TritonError",
    "MLIR",
    "mlir::",
    "_gko_gemm_epilogue",
    "post_grad_custom_pre_pass",
    "install_inductor_pass",
    "pass_fn",
)

_INSTALL_ONEARG = re.compile(
    r"install_inductor_pass\(\s*(?!['\"])(?P<fn>[^,\)\n]+?)(?:\(\))?\s*(?P<rest>,[^)]*)?\)"
)
_INSTALL_STR_FN = re.compile(
    r"install_inductor_pass\(\s*(?P<attr>['\"][^'\"]+['\"])\s*,\s*(?P<fn>['\"][^'\"]+['\"])(?P<rest>[^)]*)\)"
)
_POST_STR_FN = re.compile(
    r"install_post_grad_pass\(\s*(?P<fn>['\"][^'\"]+['\"])(?P<rest>[^)]*)\)"
)
_PASS_ATTRS = frozenset({"pre_grad_custom_pass", "post_grad_custom_pre_pass"})
_ATTN_KINDS = frozenset({"unfused_bmm", "sdpa", "attention", "windowed_attn"})
_NOT_TRITON = (
    "graph break",
    "Fullgraph",
    "fullgraph",
    "SkipFrame",
    "Unsupported: ",
    "dynamo",
    "setattr on GraphModule",
    "DataDependent",
    "PolygonMasks",
    "FakeQuant",
)


def is_triton_error(err: str, recipe: dict | None = None) -> bool:
    """True iff the crash is the generated/custom kernel, not Dynamo/region compile."""
    e = err or ""
    if "kernels/generated" in e or "kernels.generated" in e:
        return True
    if "_inductor.config" in e and "_gko_" in e:
        return True
    if any(m in e for m in _NOT_TRITON) and not any(m in e for m in ("gko::", "triton", "addmm_gelu")):
        return False
    if any(m in e for m in _TRITON_MARKERS):
        return True
    acts = (recipe or {}).get("actions") or []
    if any(a.get("op") == "replace_pattern" for a in acts) and (
        "Error" in e and ("triton" in e.lower() or "kernel" in e.lower())
    ):
        return True
    return False


def recipe_wants_generated_kernel(recipe: dict | None, scheme_name: str = "") -> bool:
    acts = list((recipe or {}).get("actions") or [])
    for a in acts:
        if a.get("op") != "replace_pattern":
            continue
        k = a.get("kernel") or ""
        if k in ("", "__generate__", "generate"):
            return True
    if scheme_name == "S6-fuse" and not any(a.get("op") == "replace_pattern" for a in acts):
        return True
    return False


def inject_kernel_id(recipe: dict, kernel_id: str) -> dict:
    out = dict(recipe)
    acts = []
    found = False
    for a in list(out.get("actions") or []):
        a = dict(a)
        if a.get("op") == "replace_pattern":
            a["kernel"] = kernel_id
            found = True
        acts.append(a)
    if not found:
        acts = [{"op": "replace_pattern", "kernel": kernel_id}] + acts
    if out.get("compile_step"):
        acts = [a for a in acts if a.get("op") != "compile"]
        out["compile_optimizer"] = False
    elif not any(a.get("op") == "compile" for a in acts):
        acts.append({"op": "compile", "path": ""})
    out["actions"] = acts
    return out


def kernel_id_from_recipe(recipe: dict | None) -> str:
    for a in (recipe or {}).get("actions") or []:
        if a.get("op") == "replace_pattern":
            return str(a.get("kernel") or "")
    return ""


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", (text or "fuse").lower()).strip("_")
    return (s or "fuse")[:40]


def _extract_source(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", t)
    if m:
        return m.group(1).strip() + "\n"
    # Unclosed fence: still take the body so validate can say missing register()
    # instead of the misleading "no python in reply".
    m2 = re.search(r"```(?:python)?\s*([\s\S]+)", t)
    if m2:
        body = m2.group(1).strip()
        if body:
            return body + "\n"
    if t.startswith("{") or t.startswith("```"):
        obj = parse_json_object(t)
        if obj and isinstance(obj.get("source"), str):
            return obj["source"]
        if obj and isinstance(obj.get("code"), str):
            return obj["code"]
    return t if "def register" in t else ""


def _spec_is_attention(spec: dict | None, scheme: str = "") -> bool:
    """True when KernelWriter is implementing leftover attention, not GEMM+act."""
    spec = spec or {}
    if scheme == "S6-sdpa" or str(spec.get("scheme") or "") == "S6-sdpa":
        return True
    blob = " ".join(
        [
            str(spec.get("compute") or ""),
            str(spec.get("replace") or ""),
            " ".join(str(m) for m in (spec.get("modules") or [])),
        ]
    ).lower()
    return any(
        t in blob
        for t in (
            "attention",
            "sdpa",
            "t5attention",
            "longformer",
            "multiheaded",
            "self-attention",
            "self attention",
        )
    )


def _validation_sites(
    sites: list | None,
    spec: dict | None = None,
    scheme: str = "",
) -> list:
    """GEMM leftover next to unfused MHA must not veto an attention kernel."""
    if _spec_is_attention(spec, scheme):
        return [{"kind": "unfused_bmm"}]
    return list(sites or [])


def _pick_pass_callable_name(src: str) -> str:
    for pat in (
        r"^def (_[A-Za-z0-9]*rewrite[A-Za-z0-9_]*)\s*\(",
        r"^def (_[A-Za-z0-9]*_pass[A-Za-z0-9_]*)\s*\(",
        r"^def (_[A-Za-z0-9]*_pattern[A-Za-z0-9_]*)\s*\(",
        r"^def (_post_grad_pass)\s*\(",
    ):
        m = re.search(pat, src, re.M)
        if m:
            return m.group(1)
    return "_gko_identity_pass"


def _ensure_pass_callable_def(src: str, name: str) -> str:
    if re.search(rf"^def {re.escape(name)}\s*\(", src, re.M):
        return src
    inject = f"\ndef {name}(gm):\n    return gm\n"
    if "def register" in src:
        return re.sub(r"\ndef register\s*\(", inject + "\ndef register(", src, count=1)
    return src + inject


def _resolve_pass_fn(src: str, fn: str) -> str:
    fn = (fn or "").strip()
    if fn.endswith("()"):
        fn = fn[:-2].strip()
    inner = fn.strip("'\"")
    quoted = bool(fn[:1] in ("'", '"'))
    if quoted or inner in _PASS_ATTRS:
        if inner and re.search(rf"^def {re.escape(inner)}\s*\(", src, re.M):
            return inner
        return _pick_pass_callable_name(src)
    return fn or _pick_pass_callable_name(src)


def _ast_is_str(node) -> bool:
    if isinstance(node, ast.Str):
        return True
    return isinstance(node, ast.Constant) and isinstance(getattr(node, "value", None), str)


def _ast_string_pass_fn(tree) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == "install_inductor_pass":
            if len(node.args) >= 2 and _ast_is_str(node.args[1]):
                return True
        if name == "install_post_grad_pass" and node.args and _ast_is_str(node.args[0]):
            return True
    return False


def _rewrite_install_inductor_pass(src: str) -> str:
    """LLM often calls install_inductor_pass(fn); the first arg must be an attr str.

    Also rewrite pass_fn=\"pre_grad_custom_pass\" (a string) to a real function.
    """

    def repl_one(m):
        fn = _resolve_pass_fn(src, m.group("fn"))
        rest = (m.group("rest") or "").strip()
        key_m = re.search(r"key\s*=\s*([^\s,)]+)", rest)
        key = key_m.group(1) if key_m else '"gko_gen"'
        return (
            f'install_inductor_pass("pre_grad_custom_pass", {fn}, key={key}); '
            f"install_post_grad_pass({fn}, key={key})"
        )

    out = _INSTALL_ONEARG.sub(repl_one, src)

    def repl_str(m):
        attr = m.group("attr")
        fn = _resolve_pass_fn(out, m.group("fn"))
        rest = m.group("rest") or ""
        key_m = re.search(r"key\s*=\s*([^\s,)]+)", rest)
        key = key_m.group(1) if key_m else '"gko_gen"'
        return f"install_inductor_pass({attr}, {fn}, key={key})"

    out = _INSTALL_STR_FN.sub(repl_str, out)

    def repl_post(m):
        fn = _resolve_pass_fn(out, m.group("fn"))
        rest = m.group("rest") or ""
        key_m = re.search(r"key\s*=\s*([^\s,)]+)", rest)
        key = key_m.group(1) if key_m else '"gko_gen"'
        return f"install_post_grad_pass({fn}, key={key})"

    out = _POST_STR_FN.sub(repl_post, out)
    name = _pick_pass_callable_name(out)
    return _ensure_pass_callable_def(out, name)


def _sanitize_source(src: str) -> str:
    """Rewrite LLM register() that setattr unknown inductor.config fields.

    Torch 2.4 ConfigModule raises AttributeError on config._gko_*.
    Also rewrite install_inductor_pass(fn) → install_inductor_pass(attr, fn).
    """
    if not src:
        return src
    out = src
    if re.search(r"(?:config|inductor_config)\._gko_", out):
        if "_PASS_INSTALLED" not in out:
            out = "_PASS_INSTALLED = False\n\n" + out
        out = re.sub(
            r"if not hasattr\(\s*(?:config|inductor_config)\s*,\s*['\"]_gko_[^'\"]+['\"]\s*\)\s*:",
            "if not _PASS_INSTALLED:",
            out,
        )
        out = re.sub(
            r"[ \t]*(?:config|inductor_config)\._gko_\w+\s*=\s*True[ \t]*\n",
            "        global _PASS_INSTALLED\n        _PASS_INSTALLED = True\n",
            out,
        )
    return _rewrite_install_inductor_pass(out)


def _validate_source(src: str, sites: list | None = None) -> str | None:
    if not src or "def register" not in src:
        return "missing def register()"
    if "torch.library.Library" in src or "Library(\"gko\"" in src or "Library('gko'" in src:
        return "use @torch.library.custom_op, not torch.library.Library"
    if "custom_op" not in src and "triton_op" not in src:
        return "missing torch.library.custom_op / triton_op"
    if "def register" in src and (
        "install_inductor_pass" not in src and "install_post_grad_pass" not in src
    ):
        return "register() must call kernels.pass_install.install_inductor_pass / install_post_grad_pass"
    if re.search(r"install_inductor_pass\(\s*(?!['\"])", src):
        return (
            "install_inductor_pass(attr, pass_fn): first arg must be an attr string "
            "e.g. install_inductor_pass('pre_grad_custom_pass', _pass, key='kid')"
        )
    if re.search(
        r"install_inductor_pass\(\s*['\"][^'\"]+['\"]\s*,\s*['\"]",
        src,
    ) or re.search(r"install_post_grad_pass\(\s*['\"]", src):
        return "install_inductor_pass / install_post_grad_pass pass_fn must be a callable, not a string"
    banned = ("subprocess", "os.system", "shutil.rmtree", "eval(", "exec(", "__import__")
    for b in banned:
        if b in src:
            return f"banned token {b}"
    if re.search(r"(?:config|inductor_config)\._gko_", src):
        return "unknown inductor.config attr (torch 2.4 ConfigModule)"
    if re.search(r"hasattr\(\s*(?:config|inductor_config)\s*,\s*['\"]_gko_", src):
        return "do not store idempotency on inductor.config"
    if re.search(
        r"@torch\.library\.register_autograd\(\s*[\"']gko::[^\"']+[\"']\s*\)",
        src,
    ) or re.search(
        r"@torch\.library\.register_autograd\(\s*[\"']gko::[^\"']+[\"']\s*,\s*setup_context",
        src,
    ):
        return (
            "torch 2.4 register_autograd needs the backward fn: "
            "torch.library.register_autograd('gko::name', backward)"
        )
    if re.search(r"\S+\s*\|\s*None", src) or re.search(r"\b(?:list|dict|tuple)\s*\[", src):
        return "Python 3.8: use Optional[T] / List[T], not T | None or list[T]"
    kinds = {s.get("kind") for s in (sites or [])}
    attn_site = bool(kinds & _ATTN_KINDS)
    gemm_site = bool(kinds & {"gemm_epilogue", "conv_epilogue"}) and not attn_site
    poi_site = bool(kinds & {"split_epilogue", "launch_bound_poi", "gemm_pointwise"}) and not attn_site
    cutlass = any(
        tok in src
        for tok in (
            "addmm_epilogue",
            "LinearCombinationGELU",
            "cutlass::",
            "gko_cutlass",
            "kernels.cutlass_gemm_attn",
        )
    )
    if gemm_site:
        if cutlass:
            pass
        elif "@triton.jit" not in src and "triton.language" not in src:
            return (
                "not fused GEMM epilogue: need CUTLASS addmm_epilogue "
                "(catalog addmm_gelu) or @triton.jit, not a Python F.gelu wrapper"
            )
        elif "tl.dot" not in src:
            return (
                "not fused GEMM epilogue: need CUTLASS LinearCombination or "
                "tl.dot + epilogue in one kernel (F.gelu/addmm wrapper is not S6)"
            )
        elif "BLOCK_M" not in src or "BLOCK_N" not in src:
            return "GEMM epilogue needs BLOCK_M/BLOCK_N constexpr tiles (TensorCore-friendly, typically >=64)"
    elif poi_site:
        if "@triton.jit" not in src and "triton.language" not in src:
            return "pointwise fuse: need @triton.jit for the leftover chain; keep vendor GEMM"
    elif kinds and not attn_site:
        if "@triton.jit" not in src and "triton.language" not in src:
            return "need @triton.jit custom kernel"
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"syntax: {e}"
    if _ast_string_pass_fn(tree):
        return "install_inductor_pass / install_post_grad_pass pass_fn must be a callable, not a string"
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        deco = " ".join(ast.dump(d) for d in node.decorator_list)
        if "custom_op" not in deco:
            continue
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation is None:
                return (
                    f"custom_op {node.name}({arg.arg}) needs a torch.Tensor annotation "
                    "(torch 2.4 infer_schema)"
                )
    return None


def write_kernel(kernel_id: str, source: str) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    init = GENERATED_DIR / "__init__.py"
    if not init.exists():
        init.write_text("")
    path = GENERATED_DIR / f"{kernel_id}.py"
    path.write_text(source if source.endswith("\n") else source + "\n")
    return path


CATALOG_KERNEL_IDS = frozenset({"fused_sdpa", "addmm_gelu", "silu_mul"})


def catalog_fallback(sites: list[dict]) -> str | None:
    for s in sites:
        fb = s.get("catalog_fallback")
        if fb:
            return str(fb)
    kinds = {s.get("kind") for s in sites}
    if "gemm_epilogue" in kinds:
        for s in sites:
            epis = [str(x).lower() for x in (s.get("epilogues") or [])]
            if s.get("kind") == "gemm_epilogue" and "gelu" in epis:
                return "addmm_gelu"
    toks: list[str] = []
    for s in sites:
        toks.extend(str(t) for t in (s.get("tokens") or []))
    if "split_epilogue" in kinds and "silu" in toks:
        return "silu_mul"
    return None


def _prefer_catalog(sites: list[dict]) -> str | None:
    """Known holes already have an insert+speedup catalog kernel. Skip LLM tl.dot."""
    fb = catalog_fallback(sites)
    if fb in ("addmm_gelu", "silu_mul", "fused_sdpa"):
        return fb
    return None


class TritonWriterAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    def generate(
        self,
        obs: Observation,
        *,
        recipe: dict | None = None,
        case_tag: str = "",
    ) -> tuple[str, str, str]:
        """Return (kernel_id, source_or_empty, llm_raw). Empty source ⇒ use catalog id."""
        from agent.kernel_spec import spec_wants_catalog

        sites = sites_from_features(obs.features)
        spec = (recipe or {}).get("kernel_spec") if isinstance(recipe, dict) else None
        catalog = spec_wants_catalog(spec)
        scheme = str((spec or {}).get("scheme") or "") if isinstance(spec, dict) else ""
        # Catalog only when the Optimizer named that hole, or there is no LLM.
        if catalog and self.llm is None:
            return catalog, "", f"catalog {catalog} (no LLM; kernel_spec.catalog_hint)"
        if self.llm is None:
            if scheme == "S6-sdpa":
                return "fused_sdpa", "", "heuristic: fused_sdpa format template (no LLM)"
            fb = catalog_fallback(sites) or catalog
            kid = fb or "addmm_gelu"
            return kid, "", f"heuristic: catalog {kid} (no LLM)"
        from agent.prompts import SYSTEM_TRITON, user_triton_generate

        attn = _spec_is_attention(spec, scheme)
        hint_kind = "sdpa" if attn else (sites[0]["kind"] if sites else "fuse")
        user = user_triton_generate(
            case=obs.model,
            sites=sites,
            features={
                "inductor_extern": obs.features.get("inductor_extern"),
                "inductor_triton": obs.features.get("inductor_triton"),
                "trace_hints": obs.features.get("trace_hints"),
                "aten_ops": obs.features.get("aten_ops"),
                "profiler": (obs.features.get("profiler") or {}).get("cuda_time_share_pct"),
            },
            kernel_id_hint=_slug(f"{obs.model}_{hint_kind}"),
            kernel_spec=spec if isinstance(spec, dict) else None,
            model_tree=(obs.model_tree or ""),
        )
        raw = self.llm.chat(SYSTEM_TRITON, user)
        src = _sanitize_source(_extract_source(raw))
        val_sites = _validation_sites(sites, spec, scheme)
        err = _validate_source(src, val_sites) if src else "no python in reply"
        if err:
            kid = _slug(case_tag or f"g_{obs.model}_{hint_kind}")
            if src:
                write_kernel(f"{kid}_rejected", src)
            fb = catalog
            if not fb and scheme == "S6-sdpa":
                fb = "fused_sdpa"
            elif not fb:
                fb = catalog_fallback(sites)
            if fb:
                return fb, "", f"generate fallback {fb}: {err}\n{raw[:1500]}"
            return kid, "", f"generate failed, no catalog: {err}\n{raw[:1500]}"
        kid = _slug(case_tag or f"g_{obs.model}_{hint_kind}")
        write_kernel(kid, src)
        return kid, src, raw

    def repair(
        self,
        obs: Observation,
        *,
        kernel_id: str,
        source: str,
        error: str,
        history: list | None = None,
    ) -> tuple[str, str, str]:
        fb = catalog_fallback(sites_from_features(obs.features))
        if kernel_id.lower() in CATALOG_KERNEL_IDS:
            fb = kernel_id
        elif not fb:
            fb = "addmm_gelu"
        if (
            error.startswith("FUSION_MISS")
            or "infer_schema" in error
            or "custom_graph_pass" in error
            or "unsupported operand type(s) for |" in error
        ):
            if kernel_id.lower() in CATALOG_KERNEL_IDS:
                return kernel_id, source, f"keep catalog {kernel_id}: {error[:500]}"
            return fb, "", f"kernel contract fail → catalog {fb}: {error[:500]}"
        if self.llm is None:
            return fb, "", "heuristic triton repair: catalog fallback"
        from agent.prompts import SYSTEM_TRITON, user_triton_repair

        path = GENERATED_DIR / f"{kernel_id}.py"
        cur = source or (path.read_text() if path.exists() else "")
        raw = self.llm.chat(
            SYSTEM_TRITON,
            user_triton_repair(
                case=obs.model,
                kernel_id=kernel_id,
                source=cur[-12000:],
                error=error[-6000:],
                history=history,
            ),
        )
        src = _sanitize_source(_extract_source(raw))
        sites = sites_from_features(obs.features)
        kid_l = (kernel_id or "").lower()
        scheme = (
            "S6-sdpa"
            if any(t in kid_l for t in ("sdpa", "attn", "attention", "longformer"))
            else ""
        )
        err = _validate_source(src, _validation_sites(sites, None, scheme)) if src else "no python in reply"
        if err:
            if src:
                write_kernel(f"{_slug(kernel_id)}_rejected", src)
            return fb, "", f"repair fallback {fb}: {err}\n{raw[:1500]}"
        write_kernel(kernel_id, src)
        return kernel_id, src, raw
