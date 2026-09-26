"""Per-scheme usage examples in memory/casememory (one markdown file each).

Hard retrieve still picks the scheme name. These files travel with that name:
Optimizer sees how the scheme was used on a real case (recipe + comments).
New collected wins are written as extra files under the same scheme.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from agent import GKO

CASEMEMORY_DIR = GKO / "memory" / "casememory"

_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
_JSON_FENCE = re.compile(r"```json\s*\n(.*?)```", re.S)
_PY_FENCE = re.compile(r"```python\s*\n(.*?)```", re.S)


def _parse_simple_yaml(text: str) -> dict:
    meta: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v.lower() in ("true", "false"):
            meta[k] = v.lower() == "true"
        else:
            try:
                if "." in v:
                    meta[k] = float(v)
                else:
                    meta[k] = int(v)
            except ValueError:
                meta[k] = v
    return meta


def parse_case_file(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {"id": path.stem, "path": str(path)}
    m = _FRONT.match(raw)
    body = raw
    if m:
        meta.update(_parse_simple_yaml(m.group(1)))
        body = raw[m.end() :]
    meta["body"] = body.strip()
    jm = _JSON_FENCE.search(body)
    if jm:
        try:
            meta["recipe"] = json.loads(jm.group(1))
        except json.JSONDecodeError:
            meta["recipe"] = None
    how = []
    grab = False
    for line in body.splitlines():
        if line.startswith("## "):
            grab = line.lower().startswith("## how") or line.lower().startswith("## 怎么")
            continue
        if grab:
            if line.startswith("## "):
                break
            how.append(line)
    meta["how"] = "\n".join(how).strip()
    # Optional Writer usage snippet (USAGE ILLUSTRATION).
    if "## writer snippet" in body.lower() or "## usage snippet" in body.lower():
        pm = _PY_FENCE.search(body)
        if pm:
            meta["writer_snippet"] = pm.group(1).strip()
    return meta


def prompt_card(entry: dict, *, limit: int = 1600) -> dict:
    how = str(entry.get("how") or entry.get("note") or "")[:limit]
    card = {
        "id": entry.get("id"),
        "scheme": entry.get("scheme"),
        "case": entry.get("case"),
        "suite": entry.get("suite") or "",
        "role": entry.get("role") or "canonical",
        "how": how,
        "recipe": entry.get("recipe"),
        "vs_naive_pct": entry.get("vs_naive_pct"),
    }
    # Optional kernel/hook snippets: USAGE ILLUSTRATIONS for Writer/Optimizer.
    snip = entry.get("writer_snippet") or entry.get("usage_snippet")
    if snip:
        card["writer_snippet"] = str(snip)[:1200]
        card["snippet_note"] = (
            "USAGE ILLUSTRATION only — adapt names/shapes/mask to THIS case; "
            "do not paste verbatim."
        )
    return card


def _md(spec: dict) -> str:
    recipe = spec.get("recipe") or {}
    recipe_s = json.dumps(recipe, indent=2, ensure_ascii=False)
    vs = spec.get("vs_naive_pct")
    vs_line = f"vs_naive_pct: {vs}\n" if vs is not None else ""
    snip = str(spec.get("writer_snippet") or spec.get("usage_snippet") or "").strip()
    snip_block = ""
    if snip:
        snip_block = (
            "\n## Writer snippet (USAGE ILLUSTRATION — adapt to THIS case)\n\n"
            f"```python\n{snip}\n```\n"
        )
    return (
        f"---\n"
        f"id: {spec['id']}\n"
        f"scheme: {spec['scheme']}\n"
        f"case: {spec['case']}\n"
        f"suite: {spec.get('suite') or ''}\n"
        f"role: {spec.get('role') or 'canonical'}\n"
        f"{vs_line}"
        f"---\n\n"
        f"# {spec['scheme']} · {spec['case']}\n\n"
        f"## How to use this scheme\n\n"
        f"{spec.get('how', '').strip()}\n"
        f"{snip_block}\n"
        f"## Recipe (apply with eval/recipe.py apply_recipe)\n\n"
        f"```json\n{recipe_s}\n```\n"
    )


CANONICAL: list[dict] = [
    {
        "id": "S1__yolov3",
        "scheme": "S1",
        "case": "yolov3",
        "suite": "torchbench",
        "vs_naive_pct": 35.0,
        "how": (
            "Use S1 when dynamo.explain(forward) shows many hot-path breaks "
            "(setattr / nonzero / Python loop) and vanilla ≈ eager (compile never fused the tower).\n"
            "yolov3: YOLOLayer.create_grids did setattr(self.nx) on the train path. "
            "rewrite_class_forward kind=yolo_layer_train_no_grid removes that, then compile path=\"\" fullgraph=True.\n"
            "Do not Titan-compile each Bottleneck (F1)."
        ),
        "recipe": {
            "model": "yolov3",
            "backend": "inductor",
            "fullgraph": False,
            "inductor": {"cudagraphs": False},
            "note": "S1: skip YOLO grid setattr, then whole Darknet fullgraph",
            "actions": [
                {"op": "rewrite_class_forward", "class": "YOLOLayer", "kind": "yolo_layer_train_no_grid"},
                {"op": "compile", "path": "", "fullgraph": True, "dynamic": False},
            ],
        },
    },
    {
        "id": "S1__speech_transformer",
        "scheme": "S1",
        "case": "speech_transformer",
        "suite": "torchbench",
        "vs_naive_pct": 5.2,
        "how": (
            "Same family as yolov3: breaks are inside pad-mask / Decoder.preprocess, not between fused layers.\n"
            "speech_vectorize_pad_mask + decoder_preprocess_tensor + no-inplace encoder/decoder + sdp_attn_no_numpy_inf, "
            "then one fullgraph compile of the whole Transformer.\n"
            "capture_scalar_outputs helps leftover .item()."
        ),
        "recipe": {
            "model": "speech_transformer",
            "backend": "inductor",
            "fullgraph": False,
            "inductor": {"cudagraphs": False},
            "dynamo": {"capture_scalar_outputs": True},
            "note": "S1 speech: vectorize pad-mask then fullgraph",
            "actions": [
                {"op": "speech_vectorize_pad_mask"},
                {"op": "rewrite_class_forward", "class": "Decoder", "kind": "decoder_preprocess_tensor"},
                {"op": "rewrite_class_forward", "class": "EncoderLayer", "kind": "encoder_layer_no_inplace"},
                {"op": "rewrite_class_forward", "class": "DecoderLayer", "kind": "decoder_layer_no_inplace"},
                {"op": "rewrite_class_forward", "class": "ScaledDotProductAttention", "kind": "sdp_attn_no_numpy_inf"},
                {"op": "compile", "path": "", "fullgraph": True, "dynamic": False},
            ],
        },
    },
    {
        "id": "S1__print_logging",
        "scheme": "S1",
        "case": "print_side_effect",
        "suite": "microbench",
        "vs_naive_pct": 0.0,
        "how": (
            "When dynamo.explain shows print/logging/warnings as graph breaks and vanilla≈eager, "
            "do not AST-edit the user's forward. Emit op allow_logging (Dynamo "
            "reorderable_logging_functions), then compile path=\"\" fullgraph=True.\n"
            "That is the closed way to remove IO graph breaks. Do not invent a rewrite_class_forward kind."
        ),
        "recipe": {
            "model": "print_side_effect",
            "backend": "inductor",
            "fullgraph": False,
            "note": "S1: reorder print/logging then fullgraph",
            "actions": [
                {"op": "allow_logging"},
                {"op": "compile", "path": "", "fullgraph": True, "dynamic": False},
            ],
        },
    },
    {
        "id": "S2__BartForCausalLM",
        "scheme": "S2",
        "case": "BartForCausalLM",
        "suite": "huggingface",
        "vs_naive_pct": 10.7,
        "how": (
            "Use S2 only when inspect prints layerdrop_n >= 1. n=0 (T5, Distil) is not S2.\n"
            "hf_layerdrop_off first (must run before compile), then compile path=\"\" fullgraph=True, "
            "and compile_optimizer=true for leftover Adam launches.\n"
            "Do not copy this JSON onto DistilBert."
        ),
        "recipe": {
            "model": "BartForCausalLM",
            "backend": "inductor",
            "fullgraph": False,
            "inductor": {"cudagraphs": False},
            "compile_optimizer": True,
            "note": "S2: hf_layerdrop_off then fullgraph + compiled Adam",
            "actions": [
                {"op": "hf_layerdrop_off"},
                {"op": "compile", "path": "", "fullgraph": True, "dynamic": False},
            ],
        },
    },
    {
        "id": "S3-unfused__BertForMaskedLM",
        "scheme": "S3-unfused",
        "case": "BertForMaskedLM",
        "suite": "huggingface",
        "vs_naive_pct": 5.6,
        "how": (
            "0-break transformer, standard Adam, vanilla/eager <= 1.20, hidden>=768, >=12 real transformer layers.\n"
            "BertLayer is already one graph; remaining gain is compiled Adam plus fullgraph to fuse leftover pointwise.\n"
            "actions: compile path=\"\" fullgraph=true AND compile_optimizer=true.\n"
            "Do not use this on DistilBert (6 layers, vanilla already 2.6x) or BERT_pytorch (vanilla already 1.2–2.2x → S3b)."
        ),
        "recipe": {
            "model": "BertForMaskedLM",
            "backend": "inductor",
            "fullgraph": False,
            "compile_optimizer": True,
            "dynamo": {"capture_scalar_outputs": True},
            "note": "S3-unfused: fullgraph + compiled Adam",
            "actions": [{"op": "compile", "path": "", "fullgraph": True, "dynamic": False}],
        },
    },
    {
        "id": "S3b__BERT_pytorch",
        "scheme": "S3b",
        "case": "BERT_pytorch",
        "suite": "torchbench",
        "vs_naive_pct": 5.0,
        "how": (
            "0-break + Adam but vanilla already 1.20–2.20× eager (or setattr/.item). "
            "fullgraph=True was −6.9% on this case.\n"
            "Use compile path=\"\" fullgraph=false + compile_optimizer=true. Do not copy BertForMaskedLM fullgraph."
        ),
        "recipe": {
            "model": "BERT_pytorch",
            "backend": "inductor",
            "fullgraph": False,
            "compile_optimizer": True,
            "note": "S3b: no fullgraph + compiled Adam",
            "actions": [{"op": "compile", "path": "", "fullgraph": False, "dynamic": False}],
        },
    },
    {
        "id": "S3-deep__XLNetLMHeadModel",
        "scheme": "S3-deep",
        "case": "XLNetLMHeadModel",
        "suite": "huggingface",
        "vs_naive_pct": 53.3,
        "how": (
            "Deep leftover (n_layer>=20 or HF_LLM) with vanilla_ms typically >=80 and Adam.\n"
            "compile_optimizer=true; extra fullgraph only if LayerDrop n>=1 (then this overlaps S2).\n"
            "HF_LLM: eval with --no-amp. Do not use on MegatronBert whose vanilla_ms is already <80 (F2-skinny)."
        ),
        "recipe": {
            "model": "XLNetLMHeadModel",
            "backend": "inductor",
            "fullgraph": False,
            "compile_optimizer": True,
            "note": "S3-deep compiled Adam; LayerDrop off if n>=1",
            "actions": [
                {"op": "hf_layerdrop_off"},
                {"op": "compile", "path": "", "fullgraph": True, "dynamic": False},
            ],
        },
    },
    {
        "id": "S3-tiny-vit__deit_tiny",
        "scheme": "S3-tiny-vit",
        "case": "deit_tiny_patch16_224.fb_in1k",
        "suite": "timm",
        "vs_naive_pct": 13.6,
        "how": (
            "ViT embed_dim<=192 (DeiT-tiny), Adam, leftover 1.2–1.8x.\n"
            "fullgraph + compile_optimizer. Do not copy onto DeiT-base / Swin / CLIP (those are already fused → identity)."
        ),
        "recipe": {
            "model": "deit_tiny_patch16_224.fb_in1k",
            "backend": "inductor",
            "fullgraph": False,
            "compile_optimizer": True,
            "note": "S3-tiny-vit fullgraph + compiled Adam",
            "actions": [{"op": "compile", "path": "", "fullgraph": True, "dynamic": False}],
        },
    },
    {
        "id": "S3-embed__MT5ForConditionalGeneration",
        "scheme": "S3-embed",
        "case": "MT5ForConditionalGeneration",
        "suite": "huggingface",
        "vs_naive_pct": 16.7,
        "how": (
            "Huge vocab (>=1e5) or params (>=200M) seq2seq, layerdrop_n=0 so not S2.\n"
            "compiled Adam (+ optional fullgraph). Do not copy onto T5/T5Small (vocab 32k / 60M)."
        ),
        "recipe": {
            "model": "MT5ForConditionalGeneration",
            "backend": "inductor",
            "fullgraph": False,
            "compile_optimizer": True,
            "note": "S3-embed compiled Adam, n=0 not S2",
            "actions": [{"op": "compile", "path": "", "fullgraph": True, "dynamic": False}],
        },
    },
    {
        "id": "S3-step__GoogleFnet",
        "scheme": "S3-step",
        "case": "GoogleFnet",
        "suite": "huggingface",
        "vs_naive_pct": 5.9,
        "how": (
            "compile_step=true with actions: [] is torch.compile(train_step): one graph over "
            "fwd+bwd+opt. Naive baseline stays compile(model). Use when leftover after "
            "compile(model) is train_step Python / opt.step / a vendor fallback (cuFFT), "
            "not a hot-path rewrite (S1) and not moco/contrastive.\n"
            "FNet FFT is the canonical vendor-fallback instance: do not fullgraph the FFT "
            "and do not only compile Adam."
        ),
        "recipe": {
            "model": "GoogleFnet",
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": "S3-step: compile(train_step), FFT stays cuFFT",
            "actions": [],
        },
    },
    {
        "id": "S3-step__GPTNeoForCausalLM",
        "scheme": "S3-step",
        "case": "GPTNeoForCausalLM",
        "suite": "huggingface",
        "vs_naive_pct": 17.6,
        "how": (
            "0-break transformer, profiler other% (Python/sync) still ~60% after compile(model).\n"
            "compile(model)+compiled Adam left that leftover; compile_step=true actions:[] wraps "
            "fwd+bwd+opt (official Dynamo boundary). Do not nest compile(model) with compile_step."
        ),
        "recipe": {
            "model": "GPTNeoForCausalLM",
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": "S3-step: compile(train_step) for leftover Python/opt after fused fwd",
            "actions": [],
        },
    },
    {
        "id": "S3-fc__vgg16",
        "scheme": "S3-fc",
        "case": "vgg16",
        "suite": "torchbench",
        "vs_naive_pct": 5.0,
        "how": (
            "Named CNN but conv_param_frac<0.3, Linear dominates, Adam, profiler other% (sync) high.\n"
            "This is leftover classifier + Adam, not a fused conv tower. compile_optimizer=true.\n"
            "vgg prefix may keep fullgraph=true; alexnet uses fullgraph=false. Do not treat as F2-cnn identity."
        ),
        "recipe": {
            "model": "vgg16",
            "backend": "inductor",
            "fullgraph": False,
            "compile_optimizer": True,
            "note": "S3-fc compiled Adam; keep fullgraph on VGG",
            "actions": [{"op": "compile", "path": "", "fullgraph": True, "dynamic": False}],
        },
    },
    {
        "id": "S3-fc__alexnet",
        "scheme": "S3-fc",
        "case": "alexnet",
        "suite": "torchbench",
        "vs_naive_pct": 13.0,
        "how": (
            "Same S3-fc family as vgg16 (conv_frac=0.04). Here fullgraph=False + compiled Adam won; "
            "fullgraph+Adam was slightly worse. Copy the flags, not VGG's fullgraph=true."
        ),
        "recipe": {
            "model": "alexnet",
            "backend": "inductor",
            "fullgraph": False,
            "compile_optimizer": True,
            "note": "S3-fc alexnet: compiled Adam, no fullgraph",
            "actions": [{"op": "compile", "path": "", "fullgraph": False, "dynamic": False}],
        },
    },
    {
        "id": "S4__detectron2_maskrcnn_r_50_c4",
        "scheme": "S4",
        "case": "detectron2_maskrcnn_r_50_c4",
        "suite": "torchbench",
        "how": (
            "Only when vanilla compile FAILED. Leave Python/dynamic/Polygon/Instances eager; compile the static backbone.\n"
            "disable_forward on the parent, compile path=backbone, skip_code_on_types for PolygonMasks/Instances/Boxes.\n"
            "If naive compile already runs (vision_maskrcnn), do not use S4 — that is identity."
        ),
        "recipe": {
            "model": "detectron2_maskrcnn_r_50_c4",
            "backend": "inductor",
            "fullgraph": False,
            "inductor": {"cudagraphs": False},
            "note": "S4: backbone only; RPN/ROI eager",
            "actions": [
                {
                    "op": "skip_code_on_types",
                    "targets": [
                        "detectron2.structures.masks.PolygonMasks.__getitem__",
                        "detectron2.structures.instances.Instances.__getitem__",
                        "detectron2.structures.boxes.Boxes.__getitem__",
                    ],
                },
                {"op": "disable_forward", "path": ""},
                {"op": "compile", "path": "backbone", "dynamic": False},
                {"op": "disable_forward", "path": "proposal_generator"},
                {"op": "disable_forward", "path": "roi_heads"},
                {"op": "disable_method", "path": "roi_heads", "name": "label_and_sample_proposals"},
            ],
        },
    },
    {
        "id": "S5-qat__resnet50_quantized_qat",
        "scheme": "S5-qat",
        "case": "resnet50_quantized_qat",
        "suite": "torchbench",
        "how": (
            "QAT: compile the fp32 GraphModule. Do not wrap every FakeQuantize (old wrap −23%).\n"
            "identity compile path=\"\" is the correct usage. compiled Adam did not help here."
        ),
        "recipe": {
            "model": "resnet50_quantized_qat",
            "backend": "inductor",
            "fullgraph": False,
            "note": "S5-qat identity fp32 GraphModule",
            "actions": [{"op": "compile", "path": ""}],
        },
    },
    {
        "id": "S5-opacus__opacus_cifar10",
        "scheme": "S5-opacus",
        "case": "opacus_cifar10",
        "suite": "torchbench",
        "how": (
            "PrivacyEngine hooks: naive compile already runs. Do not skipfiles+inner fullgraph (size-assert).\n"
            "identity compile path=\"\". If you must compile, compile inner _module and leave hooks eager "
            "(recipe kind s5_opacus); this canonical file keeps identity because disable-hooks was −1%."
        ),
        "recipe": {
            "model": "opacus_cifar10",
            "backend": "inductor",
            "fullgraph": False,
            "note": "S5-opacus identity; hooks stay eager",
            "actions": [{"op": "compile", "path": ""}],
        },
    },
    {
        "id": "S5-embed__torchrec_dlrm",
        "scheme": "S5-embed",
        "case": "torchrec_dlrm",
        "suite": "torchbench",
        "how": (
            "CombinedOptimizer and/or EmbeddingBag: do not compile the optimizer, do not fullgraph EmbeddingBag.\n"
            "identity compile path=\"\"."
        ),
        "recipe": {
            "model": "torchrec_dlrm",
            "backend": "inductor",
            "fullgraph": False,
            "note": "S5-embed: CombinedOptimizer + EmbeddingBag identity",
            "actions": [{"op": "compile", "path": ""}],
        },
    },
    {
        "id": "S5+S3__demucs",
        "scheme": "S5+S3",
        "case": "demucs",
        "suite": "torchbench",
        "how": (
            "LSTM/cuDNN is an intentional graph break (S5: do not compile LSTM into Inductor). "
            "If the step is still long and Adam leftover is high, still compile(model)+compile_optimizer.\n"
            "actions compile path=\"\" fullgraph=false, compile_optimizer=true."
        ),
        "recipe": {
            "model": "demucs",
            "backend": "inductor",
            "fullgraph": False,
            "compile_optimizer": True,
            "inductor": {"cudagraphs": False},
            "note": "S5 LSTM eager + compiled Adam",
            "actions": [{"op": "compile", "path": ""}],
        },
    },
    {
        "id": "S6-fuse__addmm_gelu_template",
        "scheme": "S6-fuse",
        "case": "_addmm_gelu_template",
        "suite": "template",
        "how": (
            "Only when leftover after official compile(train_step) can beat cuBLAS/cuDNN:\n"
            "HARD RULE — custom kernel involving GEMM: use CUTLASS OpClassTensorOp "
            "(or keep vendor cuBLAS/cuDNN GEMM and only fuse the leftover pointwise). "
            "NEVER primary-path Triton tl.dot GEMM; NEVER handwritten float/TC-less GEMM "
            "(both lose badly vs cuBLAS on A100; Titan SwiGLU microbench ~0.05–0.7×). "
            "Triton is only for pure pointwise / small non-GEMM leftovers.\n"
            "- extern GEMM + separate gelu: catalog addmm_gelu (CUTLASS Gemm+LinearCombinationGELU). "
            "Do NOT generate naive tl.dot (Bert 0.887×). Autograd must save pre-GELU z; "
            "recomputing addmm in bwd loses e2e train (Bert 0.80×).\n"
            "- Linear/FFN + SwiGLU (gate/up→silu*mul): CUTLASS GEMM+SwiGLU epilogue "
            "(cutlass_fmha / gko_cutlass_swiglu), NOT Triton gemm_swiglu tl.dot, NOT "
            "opaque silu_and_mul, NOT cuBLAS+separate Triton silu.\n"
            "- silu|mul split after vendor GEMM/conv (SE / leftover only): catalog silu_mul, "
            "keep vendor matmul. mobilenetv2_100 1.07×, mnasnet1_0 1.04×, "
            "tf_efficientnet_b0 1.03× vs official step.\n"
            "- many triton_poi_* AND leftover triton/copy is launch/HBM bound: generated Triton poi.\n"
            "NOT S6: bare cuBLAS, leftover cuDNN conv, Flash-only, fused silu_mul + leftover mm, "
            "GEMM≥60% with tiny epi, Triton/handwritten GEMM pretending to be epilogue.\n"
            "replace_pattern then compile_step=true (no compile(model)). KernelWriter / catalog "
            "must emit def register(), custom_op gko::*, register_autograd, pass_install FX "
            "so the op is inserted at the leftover site and actually called."
        ),
        "recipe": {
            "model": "_addmm_gelu_template",
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": "S6: replace_pattern addmm_gelu then compile(train_step)",
            "actions": [
                {"op": "replace_pattern", "kernel": "addmm_gelu"},
            ],
        },
    },
    {
        "id": "S6-sdpa__DistilBertForMaskedLM",
        "scheme": "S6-sdpa",
        "case": "DistilBertForMaskedLM",
        "suite": "huggingface",
        "vs_naive_pct": 16.3,
        "how": (
            "I-channel: extern bmm + triton softmax, no _scaled_dot_product_flash/efficient. "
            "Class is DistilBERT MultiHeadSelfAttention (q_lin/k_lin/v_lin), AMP fp16, S=128, head_dim=64.\n"
            "First leaf stays F2-skinny identity (S2/S3 Adam was DistilBert −68%). After identity, "
            "apply_rewrite sdpa_rewrite then compile_step=true (official train_step wrap). "
            "Do not also compile(model). In-tree F.sdpa + compile(train_step) 53.10→49.36 ms "
            "(+7.6% vs official step; ~16% vs naive compile(model)). "
            "Custom Triton fused_sdpa is not the first retrieve — vendor SDPA has the fused bwd.\n"
            "Also lands on OPTForCausalLM (causal is_causal / vendor FMHA). Longformer / T5 / "
            "4D rel-bias / windowed attn use fused_sdpa_attn + replace_pattern kernel=__generate__, "
            "not this DistilBert rewrite.\n"
            "Do not overwrite eval/recipes/DistilBertForMaskedLM.json (gold identity)."
        ),
        "recipe": {
            "model": "DistilBertForMaskedLM",
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": "S6-sdpa: MultiHeadSelfAttention -> F.sdpa then compile(train_step)",
            "actions": [
                {"op": "apply_rewrite", "rewrite": "sdpa_rewrite"},
            ],
        },
    },
    {
        "id": "S6-sdpa__OPTForCausalLM",
        "scheme": "S6-sdpa",
        "case": "OPTForCausalLM",
        "suite": "huggingface",
        "vs_naive_pct": 24.1,
        "how": (
            "Causal decoder, unfused bmm+softmax, no in-tree Flash. Hook OPTAttention "
            "q_proj/k_proj/v_proj to F.scaled_dot_product_attention(is_causal=True) "
            "(vendor CUTLASS FMHA) then compile_step=true. Official compile(train_step) "
            "54.7→42.9 ms (1.28×). Do not nest compile(model). Do not start with custom "
            "Triton or handwritten tiled CUTLASS FA (~30× slower than vendor Flash). "
            "XGLM 24-layer / GPT-Neo local attn are different; T5 4D rel-bias uses "
            "fused_sdpa_attn + replace_pattern kernel=__generate__ (KernelWriter implements the bias)."
        ),
        "recipe": {
            "model": "OPTForCausalLM",
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": "S6-sdpa: OPTAttention -> F.sdpa is_causal then compile(train_step)",
            "actions": [
                {"op": "apply_rewrite", "rewrite": "sdpa_rewrite"},
            ],
        },
    },
    {
        "id": "S6-sdpa__AllenaiLongformerBase",
        "scheme": "S6-sdpa",
        "case": "AllenaiLongformerBase",
        "suite": "huggingface",
        "vs_naive_pct": 37.0,
        "how": (
            "Sliding-window leftover on long S: official leftover is unfused bmm (softmax "
            "names often stripped). Do not copy DistilBert sdpa_rewrite. Emit kernel_spec "
            "with match_eager = local window QK/softmax/PV, hook_call = fused_sdpa_attn "
            "window arg from one_sided_attn_window_size, replace_pattern __generate__, "
            "apply_rewrite fused_sdpa_attn, compile_step=true. Landed op must be "
            "torch.ops.gko.* (not F.sdpa). Dense fused_sdpa without window is wrong.\n"
            "Writer pitfalls seen in probes: (1) FX pass must use "
            "graph = gm.graph if hasattr(gm,'graph') else gm — never gm.nodes on "
            "_GMShim; (2) install_inductor_pass(attr, callable) — pass_fn is never a "
            "string; (3) custom_op returns must .clone()."
        ),
        "writer_snippet": (
            "# USAGE ILLUSTRATION — adapt window/op name to THIS case\n"
            "def _rewrite(gm):\n"
            "    graph = gm.graph if hasattr(gm, 'graph') else gm\n"
            "    for node in list(graph.nodes):\n"
            "        ...  # match KERNEL_SPEC.fx_match only\n"
            "    return gm\n"
            "def register():\n"
            "    install_inductor_pass('pre_grad_custom_pass', _rewrite, key=KID)\n"
            "    install_post_grad_pass(_rewrite, key=KID)\n"
            "# custom_op forward: return out.clone()  # never alias inputs\n"
        ),
        "recipe": {
            "model": "AllenaiLongformerBase",
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": "S6-sdpa: generate windowed FA + fused_sdpa_attn then compile(train_step)",
            "kernel_spec": {
                "compute": "sliding-window FA: QK^T inside one_sided window, softmax, PV",
                "modules": ["LongformerSelfAttention"],
                "match_eager": (
                    "Eager local branch only: windowed QK^T, mask, fp32 softmax, dropout, P@V. "
                    "Global-attn stays on original forward."
                ),
                "hook_call": (
                    "fused_sdpa_attn → gko.<op>(q,k,v,bias,scale,causal,window); "
                    "window=one_sided_attn_window_size; return context.clone()"
                ),
                "insert": "hook forward to torch.ops.gko.<op>; keep global-attn on original",
                "catalog_hint": None,
            },
            "actions": [
                {"op": "replace_pattern", "kernel": "__generate__"},
                {"op": "apply_rewrite", "rewrite": "fused_sdpa_attn"},
            ],
        },
    },
    {
        "id": "S6-sdpa__T5Small",
        "scheme": "S6-sdpa",
        "case": "T5Small",
        "suite": "huggingface",
        "vs_naive_pct": None,
        "role": "canonical",
        "how": (
            "T5Attention leftover: unfused bmm + 4D relative-position bias + fp32 softmax. "
            "Do not copy DistilBert sdpa_rewrite (F.sdpa drops the bias). Emit kernel_spec "
            "match_eager for scores+bias→softmax→dropout→P@V with scale=1.0 (scale in q "
            "weights), modules=['T5Attention'], replace_pattern __generate__, "
            "fused_sdpa_attn, compile_step. custom_op MUST return .clone() — aliasing "
            "inputs crashes torch 2.4. Repair of alias/schema is Layer C, even if the "
            "stack passes through fused_sdpa_attn.py."
        ),
        "writer_snippet": (
            "# USAGE ILLUSTRATION — T5 rel-bias contract\n"
            "@torch.library.custom_op('gko::t5_rel_attn', mutates_args=())\n"
            "def t5_rel_attn(q, k, v, bias, scale: float, causal: int, window: int):\n"
            "    # match_eager: Q@K^T (+bias), fp32 softmax, dropout, P@V; scale usually 1.0\n"
            "    out = ...\n"
            "    return out.clone()  # required — no input alias\n"
        ),
        "recipe": {
            "model": "T5Small",
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": "S6-sdpa: generate T5 rel-bias FA + fused_sdpa_attn then compile(train_step)",
            "kernel_spec": {
                "compute": "T5 attn with 4D relative bias before softmax",
                "modules": ["T5Attention"],
                "match_eager": (
                    "After Q/K/V proj: scores=Q@K^T (scale=1.0), add 4D position_bias, "
                    "fp32 softmax+cast, dropout, P@V"
                ),
                "hook_call": "fused_sdpa_attn._call → gko.<op>; return context.clone()",
                "catalog_hint": None,
            },
            "actions": [
                {"op": "replace_pattern", "kernel": "__generate__"},
                {"op": "apply_rewrite", "rewrite": "fused_sdpa_attn"},
            ],
        },
    },
    {
        "id": "S6-sdpa__BERT_pytorch",
        "scheme": "S6-sdpa",
        "case": "BERT_pytorch",
        "suite": "torchbench",
        "vs_naive_pct": None,
        "role": "canonical",
        "how": (
            "BERT_pytorch Attention already has projected QKV [B,H,S,D]. Fuse "
            "score/softmax/PV only. Hook builds additive bias via _bias_keep_mask — "
            "your op receives bias broadcastable to scores, NOT the raw Boolean mask. "
            "kernel_spec.hook_call must say that. replace_pattern __generate__ + "
            "fused_sdpa_attn + compile_step. Do not switch to compile(model) on Repair. "
            "If crash is inside fused_sdpa_attn._bias_keep_mask under Dynamo, that is "
            "Layer B (hook helper); if gko:: op shape/rel_loss, that is Layer C."
        ),
        "writer_snippet": (
            "# USAGE ILLUSTRATION — already-projected QKV + hook bias\n"
            "# hook: bias = _bias_keep_mask(mask, bs, k_len, ...); then op(q,k,v,bias,scale,...)\n"
            "# WRONG: re-apply mask.eq(0) inside the op with a different broadcast\n"
            "def forward(q, k, v, bias, scale, causal, window):\n"
            "    scores = (q @ k.transpose(-2, -1)) * scale\n"
            "    if bias is not None:\n"
            "        scores = scores + bias  # already additive keep-mask layout\n"
            "    ...\n"
            "    return ctx.clone()\n"
        ),
        "recipe": {
            "model": "BERT_pytorch",
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": "S6-sdpa: generate BERT_pytorch FA + fused_sdpa_attn then compile(train_step)",
            "kernel_spec": {
                "compute": "masked self-attn on projected QKV",
                "modules": ["Attention", "MultiHeadedAttention"],
                "match_eager": (
                    "scores=(Q@K^T)*scale, add keep-mask bias, softmax, dropout, P@V; "
                    "return context"
                ),
                "hook_call": (
                    "_bert_pytorch_attn builds bias via _bias_keep_mask then _call; "
                    "op must accept that bias layout"
                ),
                "catalog_hint": None,
            },
            "actions": [
                {"op": "replace_pattern", "kernel": "__generate__"},
                {"op": "apply_rewrite", "rewrite": "fused_sdpa_attn"},
            ],
        },
    },
    {
        "id": "S6-fuse__mobilenetv2_100",
        "scheme": "S6-fuse",
        "case": "mobilenetv2_100",
        "suite": "timm",
        "vs_naive_pct": 6.7,
        "how": (
            "SE / MBConv leftover: silu and mul in different triton_poi names. "
            "replace_pattern kernel=__generate__ then compile_step=true (no compile(model)). "
            "KernelWriter implements THIS leftover (silu×mul if that is the hole). "
            "silu_mul.py is only the autograd format sample. Keep cuDNN conv. "
            "mobilenetv2_100 36.89→34.57 ms (1.07× vs official step); mnasnet1_0 1.04×; "
            "tf_efficientnet_b0 1.03×; mobilenet_v3_large 1.02×. "
            "Do not replace leftover cuDNN conv with CUTLASS conv."
        ),
        "recipe": {
            "model": "mobilenetv2_100",
            "backend": "inductor",
            "fullgraph": False,
            "compile_step": True,
            "note": "S6: generate leftover fuse then compile(train_step)",
            "actions": [
                {"op": "replace_pattern", "kernel": "__generate__"},
            ],
        },
    },
    {
        "id": "F2-eager__resnet18",
        "scheme": "F2-eager",
        "case": "resnet18",
        "suite": "torchbench",
        "vs_naive_pct": 18.0,
        "how": (
            "vanilla torch.compile(model) is slower than eager on a small CNN step (~10ms).\n"
            "The correct recipe is skip compile: actions: [] (NOT compile path=\"\", which is identity=vanilla).\n"
            "Do not compile_optimizer. Do not copy this skip onto mv3/densenet where vanilla is already faster."
        ),
        "recipe": {
            "model": "resnet18",
            "backend": "inductor",
            "fullgraph": False,
            "note": "F2-eager: skip compile, vanilla > eager",
            "actions": [],
        },
    },
    {
        "id": "F2-cnn__resnet50",
        "scheme": "F2-cnn",
        "case": "resnet50",
        "suite": "torchbench",
        "how": (
            "True CNN, conv_frac>=0.3, fwd 0-break: naive inductor already fused the tower.\n"
            "identity = compile path=\"\" (same as vanilla). Optional tiny fullgraph is ok; "
            "do not compile_optimizer on a true CNN. If vanilla > eager at small ms, that is F2-eager skip instead."
        ),
        "recipe": {
            "model": "resnet50",
            "backend": "inductor",
            "fullgraph": False,
            "note": "F2-cnn identity; optional fullgraph",
            "actions": [{"op": "compile", "path": "", "fullgraph": True, "dynamic": False}],
        },
    },
    {
        "id": "F2-skinny__DistilBertForMaskedLM",
        "scheme": "F2-skinny",
        "case": "DistilBertForMaskedLM",
        "suite": "huggingface",
        "how": (
            "model_type contains distil, or n_layer<=6 with vanilla already >=2.3x, or hidden in (0,256], "
            "or n_tensors>500, or n_layer>=20 but vanilla_ms<80.\n"
            "identity compile path=\"\". Old S3/S2 JSON was DistilBert 66.7→112 (−68%). "
            "Missing hidden_size is NOT skinny (that used to steal BERT_pytorch)."
        ),
        "recipe": {
            "model": "DistilBertForMaskedLM",
            "backend": "inductor",
            "fullgraph": False,
            "note": "F2-skinny identity = naive compile(model)",
            "actions": [{"op": "compile", "path": ""}],
        },
    },
    {
        "id": "F6__densenet121",
        "scheme": "F6",
        "case": "densenet121",
        "suite": "torchbench",
        "how": (
            "Many DenseLayer / _DenseLayer concat. fullgraph=True was −23%.\n"
            "identity compile path=\"\", no fullgraph."
        ),
        "recipe": {
            "model": "densenet121",
            "backend": "inductor",
            "fullgraph": False,
            "note": "F6 identity, no fullgraph",
            "actions": [{"op": "compile", "path": ""}],
        },
    },
    {
        "id": "identity__microbench",
        "scheme": "identity",
        "case": "microbench_unbacked_tolist_sum",
        "suite": "torchbench",
        "how": (
            "Fallback when no accel leaf hits (or vanilla_ms<5 / n_params<0.2).\n"
            "identity is compile path=\"\" — the same as naive torch.compile(model). "
            "Do not skip compile unless F2-eager (vanilla > eager)."
        ),
        "recipe": {
            "model": "microbench_unbacked_tolist_sum",
            "backend": "inductor",
            "fullgraph": False,
            "note": "identity fallback = naive compile(model)",
            "actions": [{"op": "compile", "path": ""}],
        },
    },
]


def seed_canonical(*, overwrite: bool = False) -> list[Path]:
    CASEMEMORY_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in CANONICAL:
        path = CASEMEMORY_DIR / f"{spec['id']}.md"
        if path.exists() and not overwrite:
            continue
        path.write_text(_md(spec), encoding="utf-8")
        written.append(path)
    readme = CASEMEMORY_DIR / "README.md"
    if overwrite or not readme.exists():
        readme.write_text(
            "# casememory\n\n"
            "One file per usage example. Frontmatter `scheme:` is the retrieval link: "
            "when the tree (or explore) picks that scheme, these cards are attached to the SchemeHit "
            "and shown to the Optimizer.\n\n"
            "- `role: canonical` — curated correct usage (seeded from eval/recipes + scheme_index).\n"
            "- `role: collected` — written by `agent/loop.py` when `--no-memory-write` is off "
            "and a round produced a usable recipe.\n\n"
            "Do not treat these as a model-name lookup table. Copy the **how** + recipe shape, "
            "then substitute the current case name.\n",
            encoding="utf-8",
        )
    return written


class CaseMemory:
    def __init__(self, entries: list[dict]):
        self.entries = entries
        self.by_scheme: dict[str, list[dict]] = {}
        for e in entries:
            sch = str(e.get("scheme") or "")
            if not sch:
                continue
            self.by_scheme.setdefault(sch, []).append(e)

    @classmethod
    def load(cls, directory: Path | None = None) -> "CaseMemory":
        directory = directory or CASEMEMORY_DIR
        seed_canonical()
        entries = []
        if directory.is_dir():
            for path in sorted(directory.glob("*.md")):
                if path.name.lower() == "readme.md":
                    continue
                try:
                    entries.append(parse_case_file(path))
                except Exception as e:
                    print(f"casememory skip {path.name}: {e}", flush=True)
        return cls(entries)

    def cards_for_scheme(self, scheme: str, *, k: int = 6) -> list[dict]:
        rows = list(self.by_scheme.get(scheme) or [])
        # Prefer cards that carry Writer usage snippets, then canonical, then id.
        rows.sort(
            key=lambda e: (
                0 if e.get("writer_snippet") or e.get("usage_snippet") else 1,
                0 if e.get("role") == "canonical" else 1,
                e.get("id") or "",
            )
        )
        return [prompt_card(e) for e in rows[:k]]

    def write_collected(
        self,
        *,
        scheme: str,
        case: str,
        suite: str,
        recipe: dict,
        how: str,
        vs_naive_pct: Optional[float] = None,
        beat_naive: bool = False,
    ) -> Path:
        CASEMEMORY_DIR.mkdir(parents=True, exist_ok=True)
        safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", case)
        safe_sch = re.sub(r"[^A-Za-z0-9_.+-]+", "_", scheme)
        path = CASEMEMORY_DIR / f"collected__{safe_sch}__{safe_case}.md"
        spec = {
            "id": f"collected__{safe_sch}__{safe_case}",
            "scheme": scheme,
            "case": case,
            "suite": suite,
            "role": "collected",
            "vs_naive_pct": vs_naive_pct,
            "how": how
            or (
                f"Collected {datetime.now().isoformat(timespec='seconds')}. "
                f"beat_naive={beat_naive}. Copy recipe shape, keep scheme {scheme}."
            ),
            "recipe": recipe,
        }
        path.write_text(_md(spec), encoding="utf-8")
        return path


@lru_cache(maxsize=1)
def load_casememory() -> CaseMemory:
    return CaseMemory.load()


def attach_casememory(hit) -> None:
    """Mutate SchemeHit with linked usage cards."""
    if hit is None:
        return
    try:
        cards = load_casememory().cards_for_scheme(hit.name)
    except Exception:
        cards = []
    hit.case_cards = cards
    if cards and not hit.examples:
        hit.examples = "; ".join(
            f"{c.get('case')} ({c.get('id')})" for c in cards[:3] if c.get("case")
        )


if __name__ == "__main__":
    paths = seed_canonical(overwrite=True)
    print("seeded", len(CANONICAL), "canonical files;", len(paths), "written")
    mem = CaseMemory.load()
    print("schemes", sorted(mem.by_scheme))
