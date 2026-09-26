"""Prompts: LLM emits eval/recipe.py JSON, not free-form torch.compile scripts."""

from __future__ import annotations

RECIPE_SCHEMA = """
Emit ONE JSON object. No markdown. This is applied by eval/recipe.py apply_recipe.

{
  "model": "<case name>",
  "backend": "inductor",
  "fullgraph": false,
  "compile_optimizer": false,
  "compile_step": false,
  "dynamo": {},
  "inductor": {},
  "note": "why this recipe",
  "actions": [ ... ]
}

Top-level:
- compile_optimizer: true → after dummy step, torch.compile(opt.step). ONLY torch.optim.Adam / AdamW.
  NEVER for SGD, RMSprop, CombinedOptimizer, DPOptimizer, FastNLP custom AdamW, *Schedule.
- compile_step: true → torch.compile(train_step) (fwd+bwd+opt). Official Dynamo wrap.
  Use with actions: [] (do not also compile(model)). Retrieve when leftover after
  compile(model) is train_step Python / opt.step / vendor fallback (FNet cuFFT),
  HF_LLM 0-break (autocast/bwd still outside the module), or 0-break fused model
  with profiler other% still high. Not for moco/contrastive or S1 rewrite cases.
- actions: [] and compile_step false and compile_optimizer false → skip compile (eager). Only when
  vanilla is slower than eager AND a clean measurement is intended (resnet18). Prefer identity
  compile path="" if skip-compile is uncertain.
- identity (same as naive) is: {"actions":[{"op":"compile","path":""}]}

Allowed action ops:
- {"op":"compile","path":"","fullgraph":false,"dynamic":false}
  path="" compiles the whole module. path="backbone" compiles a submodule.
- {"op":"compile_children","path":"encoder","only_class":["Sequential"],"fullgraph":true}
- {"op":"compile_class","names":["TransformerBlock"],"fullgraph":true}
- {"op":"disable_forward","path":"rpn"}          # leave region eager
- {"op":"disable_method","path":"","name":"capture_activations_hook"}
- {"op":"hf_layerdrop_off"}                      # ONLY if features.layerdrop_n >= 1
- {"op":"speech_vectorize_pad_mask"}
- {"op":"rewrite_class_forward","class":"YOLOLayer","kind":"yolo_layer_train_no_grid"}
  kinds: yolo_layer_train_no_grid, lstm_proj_no_flatten, encoder_layer_no_inplace,
  decoder_layer_no_inplace, decoder_preprocess_tensor, sdp_attn_no_numpy_inf
- {"op":"skip_code_on_types","targets":["detectron2.structures.masks.PolygonMasks.__getitem__"]}
- {"op":"channels_last"}                         # rarely helps CNNs
- {"op":"gcn_cached"}                            # DO NOT combine with fullgraph=True (setattr fail)
- {"op":"allow_logging"}                         # print/logging IO graph-break: Dynamo reorders, no AST edit
- {"op":"mark_dynamic","input":"input_ids","dim":1}
  optional min/max. Dict inputs: use "input" key. Tuple/list: {"op":"mark_dynamic","index":0,"dim":0}.
  Never put a HuggingFace key on tuple inputs. Apply before compile.
  Retrieve this when features.trace_hints contains many_static_shape_guards.
- {"op":"dump_trace"}                            # write traces/<case>/recipe/ (names, not Triton source)
- {"op":"replace_pattern","kernel":"__generate__"}
  Layer C only. KernelWriter writes THIS case's op. The kernel field MUST be
  __generate__. Do not put fused_sdpa / addmm_gelu / silu_mul here — those names
  are Writer format examples, not recipe actions. No extra keys (no fp32_bias,
  no backend).
- {"op":"apply_rewrite","rewrite":"sdpa_rewrite"} # DistilBert/OPT only: in-tree F.sdpa (Layer B)
- {"op":"apply_rewrite","rewrite":"fused_sdpa_attn"} # S6-sdpa insert hook (not the kernel)
- {"op":"apply_rewrite","rewrite":"__generate__"} # S1-generic ONLY: RewriteWriter apply(model)
  Do not paste Python into this JSON; do not edit the benchmark model.py on disk.

S6 kernel_spec (Optimizer fills this; KernelWriter is the only reader):
{
  "compute": "the math THIS case's kernel replaces",
  "replace": "which leftover ops / which module.forward",
  "modules": ["the attention or linear class"],
  "insert": "hook .forward and/or FX-match leftover aten → torch.ops.gko.<op>",
  "match_eager": "REQUIRED: which eager ops/region the op must equal numerically",
  "hook_call": "REQUIRED for S6-sdpa: arg order from fused_sdpa_attn._call → your op",
  "fx_match": "aten pattern to rewrite if the hook is missed",
  "layout": "ranks/dtypes",
  "case_notes": "what is unique to THIS model",
  "catalog_hint": null
}
KernelWriter does NOT invent the semantic target — it implements match_eager /
hook_call from THIS brief. If those fields are vague, the Writer will fuse the
wrong subgraph.

Scheme → actions:
- S1 generic: apply_rewrite __generate__ then compile. Catalog YOLO/LayerDrop stay rewrite_class_forward.
- S6-fuse: kernel_spec + replace_pattern kernel=__generate__ + compile_step true. No compile(model).
- S6-sdpa: kernel_spec + replace_pattern kernel=__generate__ + apply_rewrite fused_sdpa_attn
  (DistilBert/OPT: sdpa_rewrite only, no replace_pattern) + compile_step true.
Never emit g_s1 as an attention hook. Never name a catalog kernel in actions.
"""

# Three intervention layers. Scheme names retrieve WHICH layer; Optimizer fills
# the operators. Do not collapse official Dynamo wrap into a kernel, or a
# graph-break patch into a fused-op.
INTERVENTION_LAYERS = """
Three layers (a recipe may compose them; each operator belongs to exactly one):

Layer A — compile region (官方 Dynamo 瞄准，不写 .py).
  Which already-existing callable Dynamo traces. Model source unchanged.
  Ops: compile / compile_children / compile_class, compile_optimizer,
  compile_step, disable_forward / disable_method, mark_dynamic, allow_logging,
  skip (empty actions).
  Official Dynamo form is compile_step=true with actions:[] :
  torch.compile(train_step) over fwd+bwd+opt. That is S3-step / leftover after
  compile(model) is Python/opt — NOT a custom kernel and NOT a method patch.
  compile(model) is also Layer A (naive / S3b). S5 leave Flash/LSTM/cuDNN eager
  is Layer A abstention.

Layer B — program / method patch (程序级，热路径 Python 把图撕碎才动).
  Bind a new function on the live module. Never overwrite disk model.py.
  Ops: apply_rewrite, rewrite_class_forward, hf_layerdrop_off,
  speech_vectorize_pad_mask.
  Job: make a legal tensor region (LayerDrop off, YOLO setattr, print,
  .item()/nonzero, Python pad loops) OR rewrite THIS module's Python so
  Inductor/vendor can see SDPA (DistilBert/OPT sdpa_rewrite → F.sdpa).
  Not a kernel. fused_sdpa_attn is only the insert hook so Layer C is traced;
  it is not the optimization. Do not emit g_s1 as an attention "repair".

Layer C — generated kernel (算子级，必须 case 定制).
  After official wrap leftover I-channel still unfused.
  Action is always replace_pattern kernel=__generate__. KernelWriter writes
  THIS leftover (windowed FA, 4D rel-bias, already-projected QKV, …).
  fused_sdpa / addmm_gelu / silu_mul are Writer format samples, not the recipe.
  Landed op must be torch.ops.gko.* (inductor_custom). F.sdpa is Layer B, not C.

Compose: S6 = C then A compile_step. S1 = B then A. S3-step = A only.
"""

# Envelope only: JSON the harness can parse. No op whitelist, no schemes.
# Used by code_only / feedback ablations — listing allowed ops IS the knowledge.
RECIPE_ENVELOPE = """
Emit ONE JSON object and nothing else (no markdown, no chain-of-thought).
The harness applies this object (eval/recipe.py). Unknown action ops,
unknown dynamo/inductor keys, or invalid JSON make the round fail.

{
  "model": "<case name>",
  "backend": "inductor",
  "fullgraph": false,
  "compile_optimizer": false,
  "compile_step": false,
  "dynamo": {},
  "inductor": {},
  "note": "short reason",
  "actions": [{"op": "compile", "path": ""}]
}

Rules:
- {"op":"compile","path":""} is naive torch.compile(the whole module). That is
  the default. Empty actions [] means skip compile (eager) — only if you
  intend to stay uncompiled.
- Leave dynamo and inductor as {}. Do not invent config key names.
- compile_optimizer true compiles Adam/AdamW .step after a dummy step.
- compile_step true compiles the whole train step (fwd+bwd+opt).
- Do not paste Python. Do not invent a file on disk. The harness will not
  list a catalog of other action ops or named schemes.
"""

SYSTEM_CODE_ONLY = """You are given one unoptimized PyTorch training workload.
Apply torch.compile to it however you think is best. You will not receive
execution timings, graph-break traces, a scheme library, or an action catalog.
""" + RECIPE_ENVELOPE

SYSTEM_FEEDBACK = """You are given one unoptimized PyTorch training workload plus
execution feedback from naive torch.compile(model) and from your previous
rounds (timings, tracebacks, leftover traces). Use that feedback to revise
how torch.compile is applied. You will not receive a scheme library, usage
examples, or an action catalog.
""" + RECIPE_ENVELOPE

SYSTEM_OPTIMIZE = """You are a PyTorch compile optimization agent.
You receive: (1) a retrieved scheme from the project's scheme index, (2) eager/vanilla timings
and inspect features (graph breaks, leftover I-channel, architecture), (3) a sketch of the model.
Output a recipe JSON that apply_recipe can run. Follow the retrieved scheme family unless a
feature clearly contradicts it. Prefer the smallest change that can beat naive torch.compile(model).
Fill ONLY the layer the scheme names. Do not turn S3-step into a kernel, or S6 into a
graph-break rewrite.

""" + INTERVENTION_LAYERS + """
Specialize the retrieved layer for THIS leftover, not a catalog clone:
- A: which callable (model / train_step / opt.step / skip / leave vendor).
- B: which Python tears the graph; patch that method only.
- C: kernel_spec MUST name match_eager (ops to equal) + hook_call / fx_match
  (how it lands). Recipe kernel=__generate__. DistilBert/OPT S6-sdpa stays B
  (sdpa_rewrite) + A compile_step.

SCHEME USAGE EXAMPLES (when attached) are expert experience from OTHER cases:
recipe shape + optional kernel/hook snippets. They are USAGE ILLUSTRATIONS —
adapt modules / mask / window / bias to THIS observation. Do not copy another
case's model name or catalog kernel id.
""" + RECIPE_SCHEMA

SYSTEM_REPAIR_SHARED = """You are a PyTorch compile repair agent for ONE intervention layer.
The previous recipe did not land: crash or fail:rel_loss (allclose vs eager, atol=rtol=1e-2).
Do NOT retrieve a new scheme. Do NOT revert to an earlier best recipe. Do NOT change layers.
Stay on the failed layer. Keep the same model name. Output recipe JSON only.

Why no revert: the orchestrator already keeps the earlier best as the run result
and as the next optimize base. Rolling back here throws away the experiment that
just failed — Repair exists to make THAT landing work, not to undo it.

STRICTLY FORBIDDEN on every layer: deleting replace_pattern / apply_rewrite / the
attention hook so allclose passes; inventing action keys (fp32_bias, backend=math,
unknown dynamo/inductor keys); replacing fused_sdpa_attn / sdpa_rewrite with g_s1;
switching compile_step → compile(model) on an S6 recipe (that abandons the Layer C
insert contract). A recipe that "passes" by dropping the kernel is not a repair.
"""

SYSTEM_REPAIR_A = SYSTEM_REPAIR_SHARED + """
This failure is Layer A (compile region). You may only change:
  compile / compile_children / compile_class / disable_forward / disable_method /
  mark_dynamic / allow_logging / compile_step / compile_optimizer / fullgraph / skip.

Typical Layer A errors (stay here):
  fullgraph + graph break / Unsupported: setattr|.item()|nonzero; OOM on whole-model
  compile; FakeQuant / Python hooks inside the compiled region; wrong submodule path;
  compile_optimizer on non-Adam; BN / AMP wrap mismatch when only the region is wrong.

Allowed fixes: drop fullgraph; compile children not parent; leave FakeQuant/Python
eager; shrink region on OOM; keep the scheme's compile_step vs compile(model) choice.

Forbidden: replace_pattern, apply_rewrite, kernel_spec, new .py, inventing a kernel
to "fix" a region crash (that is Layer C / another scheme).
""" + RECIPE_SCHEMA

SYSTEM_REPAIR_B = SYSTEM_REPAIR_SHARED + """
This failure is Layer B (method patch / insert hook). You may only change:
  rewrite_class_forward kind, apply_rewrite __generate__ (S1), hf_layerdrop_off,
  speech_vectorize_pad_mask, skip_code_on_types, allow_logging, fullgraph on the
  compile that follows the patch.

Typical Layer B errors (stay here):
  traceback in fused_sdpa_attn.py / rewrites/generated (hook or method patch itself);
  Dynamo AssertionError on a helper imported inside the hook (_bias_keep_mask alias);
  LayerDrop / YOLO setattr / print / .item() still breaking after a wrong patch;
  DistilBert/OPT sdpa_rewrite not applied to the right class.

Keep fused_sdpa_attn / sdpa_rewrite if they were in the failed recipe — those
hooks are the insert contract so Layer C is traced, not something to rewrite away.
Generated-rewrite source is RewriteWriter's job; do not paste Python here.

Forbidden: replace_pattern, catalog kernel ids, fp32_bias, compile_optimizer on CNN,
compile(model) instead of compile_step on S6, emitting g_s1 as an attention repair.
If the crash is inside kernels/generated or "Tensors returned from custom ops" /
"Offending op: gko::…", that is Layer C — but the harness already chose B only when
the hook itself failed; do not invent a kernel JSON here.
""" + RECIPE_SCHEMA

SYSTEM_REPAIR_C = SYSTEM_REPAIR_SHARED + """
This failure is Layer C (generated kernel). You may only change kernel_spec
(compute / replace / modules / insert / match_eager / hook_call / fx_match /
layout / case_notes) so KernelWriter regenerates THIS leftover.

Typical Layer C errors (stay here — tighten match_eager / hook_call, do not drop C):
  kernels/generated/*.py crash; torch.ops.gko / gko:: schema; custom_op alias
  ("Tensors returned from custom ops… must .clone()"); install_inductor_pass /
  pass_fn string / _GMShim has no .nodes (FX must use graph = gm.graph if hasattr);
  Triton CompilationError; fail:rel_loss with inductor_custom listing your op;
  mask/bias layout mismatch vs fused_sdpa_attn._call.

Actions MUST keep:
  {"op":"replace_pattern","kernel":"__generate__"}
  and apply_rewrite fused_sdpa_attn if the failed recipe had that hook.
catalog_hint must stay null. Do not name fused_sdpa / addmm_gelu / silu_mul as
the kernel — KernelWriter already has one autograd format sample.
Keep compile_step true. No compile(model). No fp32_bias. No backend=math/auto.
Kernel crashes and rel_loss of the fused op go to KernelWriter; do not paste Triton.
Emit ONE JSON object (same envelope as the failed recipe). No markdown.
"""

SYSTEM_REPAIR = SYSTEM_REPAIR_SHARED  # tests / default header; dispatch via system_repair()


def system_repair(layer: str) -> str:
    layer = (layer or "A").strip().upper()
    if layer == "C":
        return SYSTEM_REPAIR_C
    if layer == "B":
        return SYSTEM_REPAIR_B
    return SYSTEM_REPAIR_A


def user_code_only(*, case: str, workload: dict, extra: str = "") -> str:
    return (
        f"CASE: {case}\n\n"
        f"WORKLOAD (source / tree / optimizer class only):\n{_dumps(workload, 12000)}\n"
        f"{extra}\n"
        "Return the recipe JSON now."
    )


def user_feedback(*, case: str, workload: dict, observation: dict, extra: str = "") -> str:
    return (
        f"CASE: {case}\n\n"
        f"WORKLOAD:\n{_dumps(workload, 8000)}\n\n"
        f"EXECUTION FEEDBACK (eager/vanilla + inspect + previous rounds):\n"
        f"{_dumps(observation, 12000)}\n"
        f"{extra}\n"
        "Return the recipe JSON now."
    )


def user_optimize(*, case: str, observation: dict, scheme: dict, extra: str = "") -> str:
    name = str((scheme or {}).get("name") or "")
    if name.startswith("S6"):
        tail = (
            "This scheme is Layer C then A. Fill kernel_spec for THIS leftover:\n"
            "  compute / replace / modules / insert, AND match_eager + hook_call + fx_match\n"
            "  (KernelWriter must equal THAT eager region — do not leave them vague).\n"
            "Actions: replace_pattern kernel=__generate__ (never fused_sdpa/addmm_gelu/"
            "silu_mul) then compile_step. S6-sdpa also apply_rewrite fused_sdpa_attn "
            "except DistilBert/OPT (sdpa_rewrite only)."
        )
    elif name.startswith("S1"):
        tail = (
            "This scheme is Layer B then A. Patch the Python that tears the graph, "
            "then compile. Do not emit a custom kernel."
        )
    else:
        tail = (
            "This scheme is Layer A. Only choose the compile region / skip / leave vendor. "
            "Do not emit replace_pattern or a method patch unless the scheme names it."
        )
    return (
        f"CASE: {case}\n\n"
        f"RETRIEVED SCHEME (generic family — specialize below):\n{ _dumps(scheme) }\n\n"
        f"OBSERVATION (eager/vanilla + leftover I-channel + tree):\n"
        f"{ _dumps(observation, 12000) }\n"
        f"{extra}\n"
        f"{tail}\n"
        "Return the recipe JSON now."
    )


def format_repair_history(history: list | None, *, limit: int = 5) -> str:
    """Prior crash tails so a later repair does not retry the same API mistake."""
    items = [str(x).strip() for x in (history or []) if str(x).strip()]
    if not items:
        return ""
    lines = [
        "PREVIOUS REPAIR ATTEMPTS ON THIS SCHEME (do not repeat a failing pattern):"
    ]
    for i, item in enumerate(items[-limit:], 1):
        lines.append(f"- attempt {i}: {item[-500:]}")
    if len(items) >= 2 and items[-1][-180:] == items[-2][-180:]:
        lines.append(
            "The latest error matches an earlier attempt. Change the API/approach; "
            "do not emit the same register_autograd decorator, FX pass, or recipe again."
        )
    return "\n".join(lines) + "\n\n"


def user_repair(
    *,
    case: str,
    observation: dict,
    failed_recipe: dict,
    error: str,
    history: list | None = None,
    layer: str = "A",
) -> str:
    layer = (layer or "A").strip().upper()
    allow = {
        "A": (
            "Allowed: compile path/fullgraph/compile_step/disable/mark_dynamic only. "
            "Typical: graph-break under fullgraph, wrong submodule, OOM region."
        ),
        "B": (
            "Allowed: method-patch ops only. Keep fused_sdpa_attn/sdpa_rewrite. No g_s1. "
            "Typical: crash inside fused_sdpa_attn.py / rewrites/generated helpers."
        ),
        "C": (
            "Allowed: kernel_spec fields only (esp. match_eager / hook_call / fx_match). "
            "Keep replace_pattern kernel=__generate__ and the attention hook. "
            "KernelWriter writes the .py. No fp32_bias / no compile(model). "
            "Typical: custom_op alias, gko:: schema, pass_install/_GMShim, "
            "rel_loss with inductor_custom listing your op."
        ),
    }.get(layer, "Allowed: stay on this layer.")
    return (
        f"CASE: {case}\n"
        f"FAILED LAYER: {layer}\n"
        f"{allow}\n\n"
        f"{format_repair_history(history)}"
        f"FAILED RECIPE (fix this attempt, do not abandon it):\n"
        f"{ _dumps(failed_recipe) }\n\n"
        f"ERROR (use this to locate the bug; do not revert to an older recipe):\n"
        f"{error[-6000:]}\n\n"
        f"OBSERVATION:\n{ _dumps(observation, 8000) }\n\n"
        "Return a repaired recipe JSON that stays on FAILED LAYER, keeps the "
        "landing actions, and matches eager loss. Do not revert — earlier bests "
        "are already stored by the orchestrator."
    )


SYSTEM_SIMILARITY = """You judge whether a compile scheme can transfer from similar past trials.
You see ONLY a compact fingerprint plus k nearest WINS and k nearest LOSSES.
You do NOT see model source, recipes, or the full library.
Reply ONE JSON object, no markdown:
{"transfer": true|false, "scheme": "<one of the listed schemes or omit>", "from_case": "<card case>", "why": "one sentence"}
Rules:
- Prefer a WIN card whose features match (layerdrop, conv_frac, breaks, Adam, ratio).
- NEVER transfer a scheme that appears in BANNED or as a close LOSS on a similar fp
  (that is how we avoid re-exploring a known failure).
- Never invent a scheme name that is not in SCHEMES.
- If neighbors disagree or distance is large, transfer=false.
"""


def user_similarity(
    *,
    current: dict,
    hard: str,
    neighbors: list,
    schemes: dict,
    losses: list | None = None,
    banned: list | None = None,
) -> str:
    return (
        f"HARD MATCH: {hard}\n"
        f"BANNED SCHEMES: {banned or []}\n\n"
        f"CURRENT FINGERPRINT:\n{_dumps(current, 2000)}\n\n"
        f"NEAR WINS:\n{_dumps(neighbors, 2500)}\n\n"
        f"NEAR LOSSES:\n{_dumps(losses or [], 2500)}\n\n"
        f"SCHEMES (one line each):\n{_dumps(schemes, 2000)}\n\n"
        "Return the transfer JSON now."
    )


SYSTEM_EVOLVE = """You propose ONE extra retrieval predicate so a failed accel scheme
stops matching this task, without stealing historical winners.
You see: current compact fingerprint, the failed leaf's `when`, the tree path (ids only),
and a few prototype cards of the same scheme that MUST still match the old leaf
(they must NOT match your new `when`).
Reply ONE JSON object:
{"apply": true|false, "when": { ...predicate... }, "scheme": "<existing scheme name>", "reason": "one sentence"}
Predicate DSL (AND of keys; nest with and/or/not):
  eager_ok, vanilla_ok, adam, is_llm, skinny, tiny_vit, zero_break, hot_breaks,
  leftover_python, leftover_step, has_moco,
  has_lstm, has_setattr, has_dynamic, has_fnet, has_yolo, has_speech,
  has_side_effect, needs_mark_dynamic, has_unfused_epilogue, unfused_bmm_attention,
  name_contains: ["substr"], model_type_contains: ["substr"],
  layerdrop_n_gte, conv_lt, conv_gte, lin_gte, sync_gte, n_params_gte, n_params_lt,
  n_tensors_gt, n_layer_lte, n_layer_gte, hidden_lte, hidden_gte, vocab_gte,
  vanilla_ms_lt, vanilla_ms_gte, ratio_lte, ratio_gte, fwd_breaks_gte, dense_n_gte
Rules:
- apply=true only if you can name a split key that this case has and the prototypes do not.
- scheme must be an existing name (often F2-skinny or identity).
- when MUST match the current case and MUST NOT match any prototype.
- Do not dump thresholds copied from a single ms number unless it is a documented split (e.g. n_tensors>500).
- If the miss is noise / already the right scheme, apply=false.
"""


def user_evolve(
    *,
    current: dict,
    failed_scheme: str,
    failed_when: dict,
    path: list,
    vs_naive_pct,
    prototypes: list,
    allowed_schemes: list,
) -> str:
    return (
        f"FAILED SCHEME: {failed_scheme}  vs_naive_pct={vs_naive_pct}\n"
        f"FAILED LEAF when: {_dumps(failed_when, 800)}\n"
        f"PATH: {path}\n\n"
        f"CURRENT FINGERPRINT:\n{_dumps(current, 2000)}\n\n"
        f"PROTOTYPES (must keep {failed_scheme}):\n{_dumps(prototypes, 3500)}\n\n"
        f"ALLOWED SCHEMES: {allowed_schemes}\n\n"
        "Return the evolve JSON now."
    )


SYSTEM_EXPLORE = """You pick the NEXT compile scheme to try after the current one was already evaluated.
You see: current fingerprint, schemes already tried THIS run (do not repeat them),
nearby WINS, nearby LOSSES, and a short candidate list (legal transitions).
Do NOT invent ops. Pick one candidate or stop.
Reply ONE JSON:
{"next": "<scheme or stop>", "why": "one sentence"}
Rules:
- next must be in CANDIDATES or "stop".
- Do not pick a BANNED scheme (similar tasks already lost/crashed on it).
- Do not pick a TRIED scheme.
- A close LOSS with the same scheme is a hard no.
- Stop if the candidate list is empty or only repeats a known failure.
"""


def user_explore(
    *,
    current: dict,
    current_scheme: str,
    last_verdict: str,
    vs_naive_pct,
    tried: list,
    candidates: list,
    banned: list,
    wins: list,
    losses: list,
) -> str:
    return (
        f"CURRENT SCHEME: {current_scheme}  last={last_verdict} vs_naive_pct={vs_naive_pct}\n"
        f"TRIED THIS RUN: {tried}\n"
        f"BANNED: {banned}\n"
        f"CANDIDATES: {candidates}\n\n"
        f"CURRENT FINGERPRINT:\n{_dumps(current, 2000)}\n\n"
        f"NEAR WINS:\n{_dumps(wins, 2000)}\n\n"
        f"NEAR LOSSES:\n{_dumps(losses, 2000)}\n\n"
        "Return the explore JSON now."
    )


TRITON_ONESHOT = """
ONE-SHOT from eval/kernels/addmm_gelu.py — this catalog kernel actually
registers on this host (Python 3.8 + Torch 2.4) and lands on training
Linear+GELU. Copy the autograd calling convention; do not invent a decorator.
Fwd should call CUTLASS addmm_epilogue (not naive tl.dot / F.gelu(addmm)).
save_for_backward must keep the pre-GELU tensor if you change the formula;
recomputing addmm in backward loses e2e train.

def _addmm_gelu_setup_context(ctx, inputs, output):
    bias, mat1, mat2 = inputs
    ctx.save_for_backward(bias, mat1, mat2)

def _addmm_gelu_backward(ctx, grad_output):
    bias, mat1, mat2 = ctx.saved_tensors
    z = torch.addmm(bias, mat1, mat2)
    gz = torch.ops.aten.gelu_backward.default(grad_output, z, approximate="none")
    gbias = gz.reshape(-1, gz.shape[-1]).sum(0)
    gmat1 = gz.reshape(-1, gz.shape[-1]).mm(mat2.t()).reshape(mat1.shape)
    gmat2 = mat1.reshape(-1, mat1.shape[-1]).t().mm(gz.reshape(-1, gz.shape[-1]))
    return gbias, gmat1, gmat2

@torch.library.custom_op("gko::addmm_gelu", mutates_args=())
def addmm_gelu(bias: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return _addmm_gelu_forward(bias, mat1, mat2)

@addmm_gelu.register_fake
def _(bias, mat1, mat2):
    n = mat2.shape[-1]
    return torch.empty(mat1.shape[:-1] + (n,), device=mat1.device, dtype=mat1.dtype)

# Torch 2.4: backward is positional. Decorator-only register_autograd is 2.5+.
torch.library.register_autograd(
    "gko::addmm_gelu",
    _addmm_gelu_backward,
    setup_context=_addmm_gelu_setup_context,
)
"""

SYSTEM_TRITON = """You write ONE Python module that registers a fused custom op for torch.compile.
Layer C only: implement KERNEL_SPEC for THIS leftover. Do not emit a
catalog clone, compile-region JSON, or apply(model).

Semantic target (do not invent a different fusion):
  KERNEL_SPEC.match_eager = the eager ops/region your op MUST equal numerically.
  KERNEL_SPEC.hook_call   = how fused_sdpa_attn calls you (arg order / return).
  KERNEL_SPEC.fx_match    = aten pattern if the hook is missed.
The Optimizer assigned those fields; follow them. The format sample below is
ONLY the autograd calling convention — not the math to fuse.

File: eval/kernels/generated/<id>.py. kernels.register(id) calls register()
BEFORE compile(train_step). Op namespace torch.ops.gko (inductor_custom).
Do NOT patch Dynamo/Inductor internals or write /tmp/torchinductor_*.
Do NOT lower to F.sdpa (that leaves inductor_custom empty).
Do NOT emit naive tl.dot against leftover cuBLAS.

register() must be idempotent (module-level _PASS_INSTALLED):
  @torch.library.custom_op("gko::<name>") with torch.Tensor annotations
  (Python 3.8: no T | None / list[T]; use Optional[T]).
  NEVER torch.library.Library / .define / .impl.
  torch.library.register_autograd("gko::<name>", backward, setup_context=setup)
  — positional; decorator-only register_autograd is 2.5+ and WRONG on this host.
  custom_op outputs MUST .clone() (returning an input/view → alias RuntimeError).
  FX via kernels.pass_install.install_inductor_pass + install_post_grad_pass
  (never assign a list; never setattr inductor.config._gko_*).
  RIGHT: install_inductor_pass("pre_grad_custom_pass", _pass, key="kid")
         install_post_grad_pass(_pass, key="kid")
  WRONG: install_inductor_pass(_pass)  # TypeError missing pass_fn
  WRONG: install_inductor_pass("pre_grad_custom_pass", "pre_grad_custom_pass")
         # pass_fn must be a callable, not a string
  graph = gm.graph if hasattr(gm, "graph") else gm
  WRONG: for node in gm.nodes  # _GMShim has no .nodes — use graph.nodes
  Rewrite EVERY matching site (loop, no break-after-first). Match KERNEL_SPEC.fx_match.
  If KERNEL_SPEC.modules is set, fused_sdpa_attn calls YOUR torch.ops.gko.<name>
  (q,k,v,bias,scale,causal,window) — not catalog fused_attention. Expose
  fused_attention = that op if you can.

Forbidden: eval(, exec(, subprocess, os.system, shutil.rmtree, __import__.
Numerics: CUDA fp32 allclose vs eager atol=2e-2.

ONE format sample (autograd calling convention only — implement KERNEL_SPEC math):
""" + TRITON_ONESHOT + """
Reply with a python code fence only (the full module). No recipe JSON.
"""


def user_triton_generate(
    *,
    case: str,
    sites: list,
    features: dict,
    kernel_id_hint: str,
    kernel_spec: dict | None = None,
    model_tree: str = "",
) -> str:
    spec_block = _dumps(kernel_spec or {}, 4500)
    return (
        f"CASE: {case}\n"
        f"KERNEL_ID_HINT: {kernel_id_hint}\n\n"
        f"KERNEL_SPEC (Optimizer brief — YOU MUST MATCH match_eager / hook_call):\n"
        f"{spec_block}\n\n"
        f"MODEL TREE (which modules exist):\n{(model_tree or '')[:1500]}\n\n"
        f"FUSION SITES (I-channel detector; context, not an enum of legal ops):\n{_dumps(sites, 3000)}\n\n"
        f"I-CHANNEL FEATURES (kernel names only, not Triton source):\n{_dumps(features, 2500)}\n\n"
        "Insertion contract (framework will not see the op unless all of this holds):\n"
        f"- File: eval/kernels/generated/{kernel_id_hint}.py\n"
        '- Recipe: {"op":"replace_pattern","kernel":"<id>"} AND compile_step true; do NOT also compile(model)\n'
        "- kernels.register(id) imports the module and calls register() BEFORE compile(train_step)\n"
        "- Custom op namespace: torch.ops.gko — Inductor leftover must list this name in inductor_custom\n"
        "- Implement KERNEL_SPEC.match_eager exactly; hook args follow KERNEL_SPEC.hook_call\n"
        "- register() installs FX via kernels.pass_install; graph = gm.graph if hasattr else gm\n"
        "- If KERNEL_SPEC.modules is set, fused_sdpa_attn calls your torch.ops.gko.<name>; do not use catalog fused_attention\n"
        "- install_inductor_pass FIRST ARG is the attr string; pass_fn is a callable (never a string)\n"
        "- custom_op returns must .clone() (no alias of inputs)\n"
        "- register() must be safe to call twice\n"
        "- Copy the autograd calling convention from the system format sample; do not copy a catalog pattern.\n\n"
        "Write the module now."
    )


def user_triton_repair(
    *,
    case: str,
    kernel_id: str,
    source: str,
    error: str,
    history: list | None = None,
) -> str:
    el = (error or "").lower()
    hints = []
    if "alias" in el or "tensors returned from custom ops" in el:
        hints.append(
            "Alias crash: every custom_op return must be a fresh tensor "
            "(e.g. out.clone()). Do not return an input or a view of an input."
        )
    if "_gmshim" in el or "has no attribute 'nodes'" in el:
        hints.append(
            "FX pass received a Graph shim: use "
            "graph = gm.graph if hasattr(gm, 'graph') else gm; then iterate graph.nodes. "
            "Never read gm.nodes on the shim."
        )
    if "pass_fn" in el or "install_inductor_pass" in el or "not callable" in el:
        hints.append(
            "install_inductor_pass(attr, pass_fn): first arg is the attr string, "
            "second is a function object — never the string 'pre_grad_custom_pass'."
        )
    if "size of tensor" in el or "must match the size" in el:
        hints.append(
            "Shape/broadcast mismatch vs hook layout: re-read KERNEL_SPEC.hook_call "
            "and match bias/mask ranks the hook already built (do not re-decode raw masks)."
        )
    hint_block = ("FEEDBACK HINTS:\n- " + "\n- ".join(hints) + "\n\n") if hints else ""
    return (
        f"CASE: {case}\n"
        f"KERNEL_ID: {kernel_id}\n\n"
        f"{format_repair_history(history)}"
        "CURRENT MODULE:\n```python\n"
        f"{source}\n```\n\n"
        f"TRITON / CUSTOM-OP TRACEBACK:\n{error}\n\n"
        f"{hint_block}"
        "Fix the kernel module using the traceback. Keep def register() and the gko custom op.\n"
        "If the traceback is ConfigModule / unknown inductor.config attr, use a\n"
        "module-level _PASS_INSTALLED flag and kernels.pass_install (never a list).\n"
        "If install_inductor_pass is missing pass_fn, first arg must be the attr\n"
        "string: install_inductor_pass('pre_grad_custom_pass', _pass, key=id).\n"
        "If register_autograd is missing backward, call\n"
        "torch.library.register_autograd('gko::<name>', backward) — not a one-arg decorator.\n"
        "If inductor_custom is empty, the FX pass missed THIS leftover; match KERNEL_SPEC.fx_match.\n"
        "Do not change the recipe JSON. Reply with the full corrected Python module."
    )


SYSTEM_REWRITE = """You write ONE Python module that repairs a Dynamo graph-break in memory.
Layer B only: patch THIS module's Python so Dynamo can capture a legal tensor
region. Do not write Triton / torch.ops.gko. Do not replace fused_sdpa_attn.

The file is saved as eval/rewrites/generated/<id>.py and loaded by
{"op":"apply_rewrite","rewrite":"<id>"} BEFORE torch.compile.

Required:
- def apply(model) -> str  (idempotent). Mutate the in-memory nn.Module or
  torch._dynamo.config. Return a short note.
- You MAY wrap forward with torch._dynamo.disable, skip_code, vectorize a
  Python loop into tensor ops, or register reorderable_logging_functions.
- You may NOT: write/overwrite the benchmark model's source files; use
  eval/exec/subprocess/os.system; invent a new torch.compile API.
- Prefer the smallest patch that removes the listed break_sample reasons.

Reply with a python code fence only (the full module). No recipe JSON.
"""


def user_rewrite_generate(
    *,
    case: str,
    rewrite_id_hint: str,
    break_sample: list,
    break_kinds: list,
    forward_src: str,
    model_tree: str,
) -> str:
    return (
        f"CASE: {case}\n"
        f"REWRITE_ID_HINT: {rewrite_id_hint}\n\n"
        f"BREAK_KINDS: {break_kinds}\n"
        f"BREAK_SAMPLE:\n{_dumps(break_sample, 2500)}\n\n"
        f"MODEL TREE (sketch):\n{model_tree}\n\n"
        f"EXISTING forward (read-only snapshot):\n{forward_src}\n\n"
        "Insertion contract:\n"
        f"- File: eval/rewrites/generated/{rewrite_id_hint}.py\n"
        '- Recipe: {"op":"apply_rewrite","rewrite":"<id>"} then compile\n'
        "- apply(model) must be safe to call twice\n\n"
        "Write the module now."
    )


def user_rewrite_repair(
    *,
    case: str,
    rewrite_id: str,
    source: str,
    error: str,
    history: list | None = None,
) -> str:
    return (
        f"CASE: {case}\n"
        f"REWRITE_ID: {rewrite_id}\n\n"
        f"{format_repair_history(history)}"
        "CURRENT MODULE:\n```python\n"
        f"{source}\n```\n\n"
        f"APPLY / COMPILE TRACEBACK:\n{error}\n\n"
        "Fix the rewrite module. Keep def apply(model).\n"
        "Do not change the recipe JSON. Reply with the full corrected Python module."
    )


def _dumps(obj, limit: int = 16000) -> str:
    import json

    text = json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    if len(text) > limit:
        return text[:limit] + "\n...[truncated]"
    return text
