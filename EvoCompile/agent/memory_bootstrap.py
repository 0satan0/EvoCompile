"""Initial retrieval tree + compact case library → memory/treememory.

Re-run:  python -m agent.memory_bootstrap
This overwrites the tree/schemes/cases seed. Evolve-added nodes are lost unless
you merged them. Prefer editing treememory (or letting evolve insert) after seed.
"""

from __future__ import annotations

import json
from pathlib import Path

GKO = Path(__file__).resolve().parents[1]
TREE_PATH = GKO / "memory" / "treememory"


def _leaf(nid, scheme, when, reason, recipe, do_not=None, examples="", confidence="hard"):
    return {
        "id": nid,
        "kind": "leaf",
        "when": when,
        "scheme": scheme,
        "reason": reason,
        "recipe": recipe,
        "do_not": do_not or [],
        "examples": examples,
        "confidence": confidence,
    }


def _sw(nid, when, children):
    return {"id": nid, "kind": "switch", "when": when, "children": children}


def default_schemes() -> dict:
    return {
        "B-loader": {
            "one_line": "eager failed; fix loader, do not compile",
            "recipe": {"kind": "identity", "note": "eager failed; do not compile"},
            "do_not": ["compile anything until eager runs"],
        },
        "S4": {
            "one_line": "naive compile failed; compile static tensor region, leave Python/dynamic eager",
            "recipe": {"kind": "identity", "note": "S4 repair: start from identity fullgraph=False"},
            "do_not": ["fullgraph=True on the parent that failed"],
        },
        "S5-qat": {
            "one_line": "QAT: identity on fp32 GraphModule, do not wrap FakeQuant",
            "recipe": {"kind": "identity", "note": "S5-qat identity fp32 GraphModule"},
            "do_not": ["wrap every FakeQuantize"],
        },
        "S5-opacus": {
            "one_line": "PrivacyEngine: compile inner _module, hooks eager",
            "recipe": {"kind": "s5_opacus", "note": "S5-opacus"},
            "do_not": ["skipfiles+inner fullgraph (size-assert)"],
        },
        "S3-deep": {
            "one_line": "HF_LLM or deep leftover + Adam: compile(model)+compiled Adam, no extra fullgraph",
            "recipe": {"kind": "compile_adam", "fullgraph": False, "note": "S3-deep"},
            "do_not": ["compile(model)+outer AMP if bwd dtype fails", "per-layer Titan compile"],
        },
        "F2-eager": {
            "one_line": "vanilla slower than eager at small step; skip compile",
            "recipe": {"kind": "skip_compile", "note": "F2-eager: skip compile"},
            "do_not": ["fullgraph=True", "compile_optimizer"],
        },
        "S1": {
            "one_line": "hot-path Python/setattr/nonzero/print tears the graph; repair then fullgraph",
            "recipe": {"kind": "s1_generic", "note": "S1: repair break then compile fullgraph"},
            "do_not": ["Titan per-Bottleneck compile (F1)", "AST-edit user forward"],
        },
        "S2": {
            "one_line": "layerdrop_n>=1: hf_layerdrop_off + fullgraph + compiled Adam",
            "recipe": {"kind": "s2", "note": "S2 LayerDrop"},
            "do_not": ["hf_layerdrop_off when n=0", "treat Distil/T5 as Bart"],
            "examples": "Bart +10.7%; Blenderbot +41%",
        },
        "F6": {
            "one_line": "many DenseLayer concat; identity, no fullgraph",
            "recipe": {"kind": "identity", "note": "F6 identity, no fullgraph"},
            "do_not": ["fullgraph=True"],
        },
        "S5-embed": {
            "one_line": "CombinedOptimizer / EmbeddingBag: identity, do not compile opt",
            "recipe": {"kind": "identity", "note": "S5 embedding identity"},
            "do_not": ["compile_optimizer", "fullgraph=True"],
        },
        "S3-step": {
            "one_line": "leftover is outside compile(model): wrap train_step (fwd+bwd+opt). Official Dynamo boundary.",
            "recipe": {"kind": "s3_step", "note": "S3-step compile(train_step)"},
            "do_not": [
                "nested compile(model)+compile_step",
                "fullgraph=True on FFT / vendor fallback",
                "only compile Adam when leftover is train_step Python",
                "wrap moco/contrastive",
                "replace S1 rewrite (yolov3)",
            ],
            "examples": "GoogleFnet +5.9%; GPTNeoCLM compile(train_step); HF_LLM official wrap",
        },
        "S3-fc": {
            "one_line": "named CNN but low conv_frac / high Linear+sync + Adam → compiled Adam",
            "recipe": {
                "kind": "compile_adam",
                "fullgraph_if_prefix": "vgg",
                "note": "S3-fc compiled Adam",
            },
            "do_not": ["treat as F2-cnn identity"],
            "examples": "vgg16 +5%; alexnet +13%",
        },
        "F2-cnn": {
            "one_line": "true CNN 0-break; naive already fused; identity",
            "recipe": {"kind": "identity", "note": "F2-cnn identity"},
            "do_not": ["compile_optimizer on true CNN", "channels_last (≈0%)"],
        },
        "S5": {
            "one_line": "vendor LSTM/RNN; leave eager",
            "recipe": {"kind": "identity", "note": "S5 identity; optional compile lstm.linear"},
            "do_not": ["fullgraph the LSTM"],
        },
        "S5+S3": {
            "one_line": "intentional LSTM break; still compile(model)+Adam if step is long",
            "recipe": {"kind": "compile_adam", "fullgraph": False, "note": "S5 LSTM eager + compiled Adam"},
            "do_not": ["compile LSTM into Inductor"],
        },
        "F2-skinny": {
            "one_line": "skinny/distil/too-small-step transformer; identity not S3",
            "recipe": {"kind": "identity", "note": "F2-skinny identity"},
            "do_not": ["copy Bart/S2 JSON", "compiled Adam on DistilBert"],
        },
        "S3-tiny-vit": {
            "one_line": "ViT embed<=192 leftover; fullgraph+compiled Adam",
            "recipe": {"kind": "compile_adam", "fullgraph": True, "note": "S3-tiny-vit"},
            "do_not": ["copy to DeiT-base/Swin/CLIP"],
        },
        "S3-embed": {
            "one_line": "huge vocab or params; compiled Adam",
            "recipe": {"kind": "compile_adam", "fullgraph": False, "note": "S3-embed"},
            "do_not": ["same JSON on T5 vocab 32k / 60M"],
        },
        "S3-unfused": {
            "one_line": "0-break Adam leftover <=1.20x hidden>=768; fullgraph+Adam",
            "recipe": {"kind": "compile_adam", "fullgraph": True, "note": "S3-unfused"},
            "examples": "Bert 65/59/56 +5.6%",
        },
        "S3b": {
            "one_line": "0-break but vanilla already 1.2-2.2x or setattr/.item; no fullgraph + Adam",
            "recipe": {"kind": "compile_adam", "fullgraph": False, "note": "S3b no fullgraph"},
            "do_not": ["fullgraph=True (BERT_pytorch -6.9%)"],
        },
        "identity": {
            "one_line": "no accel scheme; naive compile(model)",
            "recipe": {"kind": "identity", "note": "identity fallback"},
            "do_not": ["S3", "fullgraph"],
        },
        "S6-fuse": {
            "one_line": (
                "custom kernel only when leftover can beat official: "
                "CUTLASS tensor-op GEMM+act (never Triton/handwritten GEMM) / "
                "silu_mul after vendor GEMM / launch-HBM poi; compile(train_step)"
            ),
            "recipe": {"kind": "s6_fuse", "note": "S6 generate/catalog fused kernel then compile_step"},
            "do_not": [
                "patch /tmp/torchinductor_* output_code.py",
                "treat leftover cuBLAS as a site",
                "GEMM-dominated step with tiny triton/copy",
                "replace Flash/cuDNN with naive tl.dot",
                "primary-path Triton tl.dot GEMM (use CUTLASS OpClassTensorOp)",
                "handwritten float or TC-less CUDA GEMM",
                "opaque silu_and_mul / cuBLAS+separate Triton silu as fake epilogue",
                "replace leftover cuDNN conv with CUTLASS conv",
                "nest compile(model) with compile_step",
                "one missed_gemm_epilogue flag with no I-channel names",
                "replace DistilBERT unfused bmm+softmax with custom Triton first",
            ],
        },
        "S6-sdpa": {
            "one_line": "unfused bmm+softmax: hook attention to vendor FMHA / tiled FA, then compile(train_step)",
            "recipe": {"kind": "s6_sdpa", "note": "S6-sdpa fused attention rewrite + compile_step"},
            "do_not": [
                "custom Triton fused_sdpa first on DistilBert (vendor SDPA has fused bwd)",
                "copy DistilBert sdpa_rewrite onto Longformer/T5/GPT-Neo as the only rewrite",
                "compiled Adam on DistilBert (F2-skinny −68%)",
                "overwrite F2-skinny identity as the first DistilBert leaf",
                "nest compile(model) with compile_step",
                "handwritten tiled CUTLASS FA (still ~30× vs vendor Flash)",
            ],
        },
    }


def default_tree() -> dict:
    return {
        "id": "root",
        "kind": "switch",
        "when": {},
        "children": [
            _leaf(
                "B-loader",
                "B-loader",
                {"eager_ok": False},
                "eager failed; fix eval loader, do not write a compile recipe",
                {"kind": "identity", "note": "eager failed; do not compile"},
                ["compile anything until eager runs"],
            ),
            _sw(
                "vanilla_fail",
                {"vanilla_ok": False},
                [
                    _leaf(
                        "S4-detectron",
                        "S4",
                        {"name_contains": ["detectron", "maskrcnn"]},
                        "naive compile failed on detection meta-arch; compile backbone only",
                        {"kind": "s4_detectron", "note": "S4: backbone only"},
                        ["compile ROI with dynamic=True", "trace Polygon/Instances"],
                        "detectron2_maskrcnn_r_50_c4 91→85.9",
                    ),
                    _leaf(
                        "S5-qat",
                        "S5-qat",
                        {"name_contains": ["qat", "quantized"]},
                        "QAT: compile fp32 GraphModule, do not wrap every FakeQuant",
                        {"kind": "identity", "note": "S5-qat identity fp32 GraphModule"},
                        ["wrap every FakeQuantize"],
                    ),
                    _leaf(
                        "S5-opacus",
                        "S5-opacus",
                        {"name_contains": ["opacus"]},
                        "PrivacyEngine hooks: compile inner _module, hooks eager",
                        {"kind": "s5_opacus", "note": "S5-opacus"},
                        ["skipfiles+inner fullgraph (size-assert)"],
                    ),
                    _leaf(
                        "S4-generic",
                        "S4",
                        {},
                        "naive compile failed; leave Python/dynamic eager, compile static tensor region",
                        {"kind": "identity", "note": "S4 repair: start from identity fullgraph=False"},
                        ["fullgraph=True on the parent that failed"],
                    ),
                ],
            ),
            _leaf(
                "S3-step-llm",
                "S3-step",
                {"and": [{"is_llm": True}, {"adam": True}, {"zero_break": True}]},
                "HF_LLM 0-break: leftover after compile(model) is autocast/bwd/opt; official wrap is compile(train_step)",
                {"kind": "s3_step", "note": "S3-step HF_LLM compile(train_step)"},
                [
                    "compile(model)+outer AMP if bwd dtype fails",
                    "per-layer Titan compile",
                    "nested compile(model)+compile_step",
                ],
                "Qwen/Llama official Dynamo compile(train_step); S3-deep compile_adam is explore-from",
            ),
            _leaf(
                "F2-eager",
                "F2-eager",
                {
                    "and": [
                        {"vanilla_gt_eager": True},
                        {
                            "or": [
                                {"vanilla_ms_lt": 15},
                                {"and": [{"conv_gte": 0.3}, {"vanilla_ms_lt": 40}]},
                            ]
                        },
                    ]
                },
                "vanilla slower than eager at small step; skip compile or identity (drop fullgraph)",
                {"kind": "skip_compile", "note": "F2-eager: skip compile"},
                ["fullgraph=True", "compile_optimizer"],
                "resnet18 11.7→9.97 skip compile",
            ),
            _sw(
                "S1-hot",
                {"hot_breaks": True},
                [
                    _leaf(
                        "S1-yolo",
                        "S1",
                        {"has_yolo": True},
                        "hot-path setattr/nonzero tears the tower; repair then fullgraph",
                        {"kind": "s1_yolo", "note": "S1 yolo"},
                        ["compile each Darknet layer separately"],
                        "yolov3 68.8→44.7 +35%",
                    ),
                    _leaf(
                        "S1-speech",
                        "S1",
                        {"has_speech": True},
                        "data-dependent pad-mask / slice; vectorize then fullgraph",
                        {"kind": "s1_speech", "note": "S1 speech"},
                        ["compile TransformerOptimizer"],
                    ),
                    _leaf(
                        "S1-logging",
                        "S1",
                        {"has_side_effect": True},
                        "print/logging/IO graph-break; allow_logging then fullgraph",
                        {"kind": "s1_logging", "note": "S1: reorder print/logging then fullgraph"},
                        ["AST-edit user forward to delete print", "Titan per-Bottleneck compile (F1)"],
                    ),
                    _leaf(
                        "S1-generic",
                        "S1",
                        {},
                        "fwd many graphs/breaks and vanilla≈eager; repair hot-path Python",
                        {"kind": "s1_generic", "note": "S1: repair break then compile fullgraph"},
                        ["Titan per-Bottleneck compile (F1)"],
                    ),
                ],
            ),
            _leaf(
                "S2",
                "S2",
                {"layerdrop_n_gte": 1},
                "layerdrop_n>=1; hf_layerdrop_off + fullgraph + compiled Adam",
                {"kind": "s2", "note": "S2 LayerDrop"},
                ["hf_layerdrop_off when n=0", "treat Distil/T5 as Bart"],
                "Bart +10.7%; Blenderbot +41%",
            ),
            _sw(
                "dense-concat",
                {"dense_n_gte": 12},
                [
                    _leaf(
                        "S3-step-dense",
                        "S3-step",
                        {"leftover_step": True},
                        "DenseLayer concat: do not fullgraph; leftover Python/opt is compile(train_step)",
                        {"kind": "s3_step", "note": "S3-step wrap train_step, no fullgraph on DenseLayer"},
                        ["fullgraph=True", "nested compile(model)+compile_step"],
                        "densenet wrap-gap vs identity compile(model)",
                    ),
                    _leaf(
                        "F6",
                        "F6",
                        {},
                        "DenseLayer concat; fullgraph regresses",
                        {"kind": "identity", "note": "F6 identity, no fullgraph"},
                        ["fullgraph=True"],
                        "densenet121 −23%",
                    ),
                ],
            ),
            _leaf(
                "S5-embed",
                "S5-embed",
                {"or": [{"has_combined": True}, {"has_embedbag": True}]},
                "CombinedOptimizer / EmbeddingBag: identity, do not compile opt",
                {"kind": "identity", "note": "S5 embedding identity"},
                ["compile_optimizer", "fullgraph=True"],
            ),
            _leaf(
                "S3-step",
                "S3-step",
                {"or": [{"has_fnet": True}, {"leftover_step": True}]},
                "leftover after compile(model) is train_step Python / opt.step / vendor fallback; compile(train_step)",
                {"kind": "s3_step", "note": "S3-step compile(train_step)"},
                [
                    "fullgraph=True on FFT / vendor fallback",
                    "only compile Adam when leftover is train_step Python",
                    "wrap moco/contrastive",
                    "replace S1 rewrite (yolov3)",
                    "nested compile(model)+compile_step",
                ],
                "GoogleFnet 60.0→56.4 +5.9%; GPTNeoCLM compile(train_step); 0-break high-Python leftover",
            ),
            _leaf(
                "S3-fc-head",
                "S3-fc",
                {
                    "and": [
                        {"conv_lt": 0.3},
                        {"lin_gte": 0.5},
                        {"adam": True},
                        {"sync_gte": 40},
                        {"n_params_gte": 10},
                    ]
                },
                "named CNN but low conv_frac / high Linear+sync + Adam → compiled Adam",
                {
                    "kind": "compile_adam",
                    "fullgraph_if_prefix": "vgg",
                    "note": "S3-fc compiled Adam",
                },
                ["treat as F2-cnn identity"],
                "vgg16 +5%; alexnet +13%; demucs +2% (LSTM stays cuDNN)",
            ),
            _sw(
                "cnn-0break",
                {"and": [{"conv_gte": 0.3}, {"zero_break": True}]},
                [
                    _leaf(
                        "S3-fc-cnn-sync",
                        "S3-fc",
                        {"and": [{"adam": True}, {"sync_gte": 70}, {"vanilla_ms_gte": 80}]},
                        "CNN 0-break but profiler other high on a long step; try compiled Adam",
                        {"kind": "compile_adam", "fullgraph": False, "note": "CNN with high sync: compiled Adam"},
                        ["Titan Sequential split", "compile LSTM"],
                        "demucs 406→398; Super_SloMo keep Adam",
                    ),
                    _leaf(
                        "S6-fuse-cnn",
                        "S6-fuse",
                        {"has_unfused_epilogue": True},
                        "CNN leftover after official compile(train_step): silu_mul / launch-HBM poi; catalog kernel then compile_step",
                        {"kind": "s6_fuse", "note": "S6 catalog/generate fused kernel then compile_step"},
                        [
                            "naive tl.dot vs cuBLAS",
                            "CUTLASS conv vs cuDNN",
                            "nest compile(model)+compile_step",
                            "drop kernel to pass rel_loss",
                        ],
                        "silu_mul: mobilenetv2_100 1.07× vs official step",
                    ),
                    _leaf(
                        "F2-cnn",
                        "F2-cnn",
                        {},
                        "conv_frac high 0-break CNN; naive already fused. identity",
                        {"kind": "identity", "note": "F2-cnn identity"},
                        ["compile_optimizer on true CNN", "channels_last (≈0%)"],
                    ),
                ],
            ),
            _sw(
                "lstm",
                {"or": [{"has_lstm": True}, {"has_rnn": True}, {"has_cudnn": True}]},
                [
                    _leaf(
                        "S5+S3",
                        "S5+S3",
                        {"and": [{"adam": True}, {"vanilla_ms_gte": 80}]},
                        "intentional LSTM break; still compile(model)+Adam if step is long",
                        {"kind": "compile_adam", "fullgraph": False, "note": "S5 LSTM eager + compiled Adam"},
                        ["compile LSTM into Inductor", "Titan per-Sequential"],
                    ),
                    _leaf(
                        "S5",
                        "S5",
                        {},
                        "vendor LSTM/RNN; leave eager, maybe compile projection Linear",
                        {"kind": "identity", "note": "S5 identity; optional compile lstm.linear"},
                        ["fullgraph the LSTM"],
                    ),
                ],
            ),
            _sw(
                "adam-zero-break",
                {"and": [{"adam": True}, {"zero_break": True}]},
                [
                    _leaf(
                        "F2-skinny",
                        "F2-skinny",
                        {"skinny": True},
                        "skinny/distil/too-small-step transformer; identity not S3",
                        {"kind": "identity", "note": "F2-skinny identity"},
                        ["copy Bart/S2 JSON", "compiled Adam on DistilBert"],
                        "DistilBert S3 −68%; Electra +1.6% keep if already tiny win",
                    ),
                    _leaf(
                        "S6-sdpa",
                        "S6-sdpa",
                        {"unfused_bmm_attention": True},
                        "unfused bmm+softmax leftover (no flash ATen); hook attention + compile(train_step)",
                        {"kind": "s6_sdpa", "note": "S6-sdpa fused attention rewrite + compile_step"},
                        [
                            "custom Triton fused_sdpa first on DistilBert",
                            "compiled Adam on DistilBert",
                            "nest compile(model)+compile_step",
                            "handwritten tiled CUTLASS FA",
                            "drop kernel/rewrite to pass rel_loss",
                        ],
                        "OPT vendor FMHA 1.28×; Longformer 1.37×; DistilBert explore after F2-skinny",
                    ),
                    _leaf(
                        "S6-fuse-attn",
                        "S6-fuse",
                        {"has_unfused_epilogue": True},
                        "Linear+GELU / leftover poi after official wrap; CUTLASS addmm_gelu or generated Triton then compile_step",
                        {"kind": "s6_fuse", "note": "S6 catalog/generate fused kernel then compile_step"},
                        [
                            "naive tl.dot vs cuBLAS",
                            "nest compile(model)+compile_step",
                            "drop kernel to pass rel_loss",
                        ],
                        "CUTLASS GEMM+GELU on leftover addmm+gelu",
                    ),
                    _leaf(
                        "S3-tiny-vit",
                        "S3-tiny-vit",
                        {"tiny_vit": True},
                        "ViT embed<=192 leftover; fullgraph+compiled Adam",
                        {"kind": "compile_adam", "fullgraph": True, "note": "S3-tiny-vit"},
                        ["copy to DeiT-base/Swin/CLIP"],
                        "deit_tiny +13.6%",
                    ),
                    _leaf(
                        "S3-embed",
                        "S3-embed",
                        {"or": [{"vocab_gte": 100000}, {"n_params_gte": 200}]},
                        "large vocab or params; compiled Adam",
                        {"kind": "compile_adam", "fullgraph": False, "note": "S3-embed"},
                        ["same JSON on T5 vocab 32k / 60M"],
                        "MT5 +17%",
                    ),
                    _leaf(
                        "S3-deep",
                        "S3-deep",
                        {"and": [{"n_layer_gte": 20}, {"vanilla_ms_gte": 80}]},
                        "deep leftover + long step; compiled Adam",
                        {"kind": "compile_adam", "fullgraph": False, "note": "S3-deep"},
                        ["MegatronBert 24 layers but vanilla 57ms"],
                    ),
                    _leaf(
                        "S3-unfused",
                        "S3-unfused",
                        {"and": [{"ratio_lte": 1.20}, {"hidden_gte": 768}, {"n_layer_gte": 12}]},
                        "0-break Adam leftover <=1.20x hidden>=768; fullgraph+Adam",
                        {"kind": "compile_adam", "fullgraph": True, "note": "S3-unfused"},
                        [],
                        "Bert 65/59/56 +5.6%",
                    ),
                    _leaf(
                        "S3b-item",
                        "S3b",
                        {
                            "or": [
                                {"and": [{"ratio_gte": 1.20}, {"ratio_lte": 2.20}]},
                                {"has_item": True},
                                {"has_setattr": True},
                            ]
                        },
                        "0-break but vanilla already 1.2-2.2x or setattr/.item; fullgraph=False + Adam",
                        {"kind": "compile_adam", "fullgraph": False, "note": "S3b no fullgraph"},
                        ["fullgraph=True (BERT_pytorch −6.9%)"],
                        "BERT_pytorch +10.9%; GPT2-cls +9.4%",
                    ),
                    _leaf(
                        "S3b-default",
                        "S3b",
                        {"adam": True},
                        "default leftover transformer + Adam: compile(model)+compiled Adam, no extra fullgraph",
                        {"kind": "compile_adam", "fullgraph": False, "note": "S3b default"},
                    ),
                ],
            ),
            _leaf(
                "identity-micro",
                "identity",
                {"or": [{"vanilla_ms_lt": 5}, {"n_params_lt": 0.2}]},
                "microbench / GNN / RL actor ~ms; identity",
                {"kind": "identity", "note": "small-step identity"},
                ["S3", "fullgraph"],
            ),
            _leaf(
                "S6-sdpa",
                "S6-sdpa",
                {"unfused_bmm_attention": True},
                "unfused bmm+softmax leftover (no flash ATen); hook attention + compile(train_step)",
                {"kind": "s6_sdpa", "note": "S6-sdpa fused attention rewrite + compile_step"},
                [
                    "custom Triton fused_sdpa first on DistilBert",
                    "compiled Adam on DistilBert",
                    "nest compile(model)+compile_step",
                    "handwritten tiled CUTLASS FA",
                    "drop kernel/rewrite to pass rel_loss",
                ],
                "OPT vendor FMHA 54.7→42.9 +1.28×; Longformer 1.37× vs official step",
            ),
            _leaf(
                "S6-fuse",
                "S6-fuse",
                {"has_unfused_epilogue": True},
                "I-channel leftover: CUTLASS GEMM+GELU / silu_mul / generated Triton poi; compile(train_step)",
                {"kind": "s6_fuse", "note": "S6 catalog/generate fused kernel then compile_step"},
                [
                    "patch inductor cache",
                    "S5 abstention",
                    "naive tl.dot vs cuBLAS",
                    "CUTLASS conv vs cuDNN",
                    "nest compile(model)+compile_step",
                ],
                "silu_mul: mobilenetv2_100 1.07×, mnasnet1_0 1.04×, efficientnet_b0 1.03× vs official step",
            ),
            _leaf(
                "identity-fallback",
                "identity",
                {},
                "no scheme matched; identity = naive compile(model)",
                {"kind": "identity", "note": "identity fallback"},
                [],
                "",
                "low",
            ),
        ],
    }


def _card(
    case,
    suite,
    scheme,
    beat,
    vs,
    eager,
    vanilla,
    agent,
    **fp,
):
    fp.setdefault("case", case)
    fp.setdefault("suite", suite)
    fp.setdefault("eager_ms", eager)
    fp.setdefault("vanilla_ms", vanilla)
    return {
        "case": case,
        "suite": suite,
        "scheme": scheme,
        "beat_naive": beat,
        "vs_naive_pct": vs,
        "eager_ms": eager,
        "vanilla_ms": vanilla,
        "agent_ms": agent,
        "source": "seed",
        "fp": fp,
    }


def default_cases() -> list:
    """Compact cards only. Enough for kNN; not the full bench table."""
    return [
        _card(
            "BartForCausalLM", "huggingface", "S2", True, 10.7, 69.1, 59.9, 53.5,
            adam=True, layerdrop_n=1, fwd_graphs=18, fwd_breaks=17, has_dynamic=True,
            zero_break=False, arch_class="transformer_base", n_layer=12, n_params_m=254,
            hidden=1024, ratio=1.15, model_type="bart",
        ),
        _card(
            "BlenderbotForCausalLM", "huggingface", "S2", True, 40.7, 157.1, 143.7, 85.2,
            adam=True, layerdrop_n=1, fwd_graphs=30, fwd_breaks=29, has_dynamic=True,
            zero_break=False, arch_class="transformer_base", n_layer=24, n_params_m=2500,
            ratio=1.09, model_type="blenderbot",
        ),
        _card(
            "XGLMForCausalLM", "huggingface", "S2", True, 37.1, 89.5, 77.4, 48.7,
            adam=True, layerdrop_n=1, fwd_graphs=30, fwd_breaks=29, has_dynamic=True,
            vocab_size=256000, n_layer=24, ratio=1.16, model_type="xglm",
        ),
        _card(
            "openai/whisper-tiny", "huggingface", "S2", True, 30.3, 32.8, 31.5, 22.0,
            adam=True, layerdrop_n=2, fwd_graphs=13, fwd_breaks=12, has_dynamic=True,
            n_params_m=37, ratio=1.04, model_type="whisper",
        ),
        _card(
            "M2M100ForConditionalGeneration", "huggingface", "S2", True, 22.3, 100.0, 82.9, 64.4,
            adam=True, layerdrop_n=2, fwd_graphs=31, fwd_breaks=30, has_dynamic=True,
            n_params_m=484, ratio=1.21, model_type="m2m_100",
        ),
        _card(
            "BertForMaskedLM", "huggingface", "S3-unfused", True, 5.6, 65.0, 59.0, 55.7,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            hidden=768, n_layer=12, n_tensors=199, n_params_m=110, ratio=1.10,
            conv_param_frac=0.0, linear_param_frac=0.8, sync=46, model_type="bert",
            arch_class="transformer_base",
        ),
        _card(
            "meta-llama/Llama-3.2-1B", "huggingface", "S3-deep", True, 46.5, 107.3, 81.4, 43.6,
            adam=True, is_llm=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            n_layer=16, n_params_m=1000, ratio=1.32, model_type="llama",
        ),
        _card(
            "google/gemma-2-2b", "huggingface", "S3-deep", True, 47.0, 215.2, 164.6, 87.3,
            adam=True, is_llm=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            n_layer=26, n_params_m=2600, ratio=1.31, model_type="gemma2",
        ),
        _card(
            "MT5ForConditionalGeneration", "huggingface", "S3-embed", True, 16.7, 82.7, 47.7, 39.7,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            vocab_size=250112, n_params_m=300, n_layer=16, ratio=1.73, model_type="mt5",
        ),
        _card(
            "GoogleFnet", "huggingface", "S3-step", True, 5.9, 91.6, 60.0, 56.4,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            n_layer=12, ratio=1.53, model_type="fnet", has_fnet=True, leftover_step=True,
            leftover_python=True, sync=60,
        ),
        _card(
            "GPTNeoForCausalLM", "huggingface", "S3-step", True, 17.6, 403.8, 234.4, 193.2,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            n_layer=24, n_params_m=350, ratio=1.72, model_type="gpt_neo", leftover_step=True,
            leftover_python=True, sync=60, vanilla_ms=234.4,
        ),
        _card(
            "GPT2ForSequenceClassification", "huggingface", "S3b", True, 9.3, 49.4, 37.9, 34.4,
            adam=True, layerdrop_n=0, fwd_graphs=4, fwd_breaks=3, has_setattr=True,
            zero_break=False, n_layer=12, ratio=1.30, model_type="gpt2",
        ),
        _card(
            "BERT_pytorch", "torchbench", "S3b", True, 10.9, 42.6, 31.3, 27.9,
            adam=True, layerdrop_n=0, n_layer=12, ratio=1.36, has_setattr=True,
            arch_class="transformer_base",
        ),
        _card(
            "DistilBertForMaskedLM", "huggingface", "F2-skinny", False, 0.0, 176.7, 66.7, 66.7,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            n_layer=6, ratio=2.65, model_type="distilbert", skinny=True, sync=64,
            arch_class="transformer_base",
        ),
        _card(
            "MegatronBertForCausalLM", "huggingface", "F2-skinny", False, 0.0, 92.2, 57.2, 57.2,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            n_layer=24, vanilla_ms=57.2, ratio=1.61, skinny=True, model_type="megatron-bert",
        ),
        _card(
            "MobileBertForMaskedLM", "huggingface", "F2-skinny", False, 0.0, 153.3, 87.8, 87.8,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            n_layer=24, n_tensors=1117, skinny=True, model_type="mobilebert",
        ),
        _card(
            "T5ForConditionalGeneration", "huggingface", "identity", False, 3.8, 97.1, 62.0, 59.6,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            vocab_size=32128, n_params_m=60, n_layer=12, ratio=1.57, model_type="t5",
        ),
        _card(
            "deit_tiny_patch16_224.fb_in1k", "timm", "S3-tiny-vit", True, 13.6, 32.3, 23.9, 20.7,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            hidden=192, n_layer=12, n_params_m=5.7, conv_param_frac=0.026, ratio=1.35,
            arch_class="vit_tiny", tiny_vit=True,
        ),
        _card(
            "deit_base_distilled_patch16_224", "timm", "F2-cnn", False, 4.1, 64.0, 51.0, 48.9,
            adam=True, hidden=768, n_layer=12, conv_param_frac=0.0, ratio=1.25,
            arch_class="vit_base", zero_break=True, fwd_breaks=0,
        ),
        _card(
            "vgg16", "torchbench", "S3-fc", True, 5.1, 103.5, 85.7, 81.4,
            adam=True, conv_param_frac=0.08, linear_param_frac=0.9, sync=80,
            n_params_m=138, n_tensors=32, fwd_breaks=0, zero_break=True, ratio=1.21,
            arch_class="other",
        ),
        _card(
            "alexnet", "torchbench", "S3-fc", True, 12.7, 14.4, 11.8, 10.4,
            adam=True, conv_param_frac=0.1, linear_param_frac=0.85, sync=50,
            n_params_m=61, fwd_breaks=0, zero_break=True, ratio=1.22, arch_class="cnn",
        ),
        _card(
            "resnet50", "torchbench", "F2-cnn", False, 1.6, 36.0, 33.1, 32.6,
            adam=False, opt="SGD", conv_param_frac=0.9, linear_param_frac=0.05,
            fwd_breaks=0, zero_break=True, n_params_m=25, arch_class="cnn", ratio=1.09,
        ),
        _card(
            "resnet18", "torchbench", "F2-eager", True, 14.7, 9.9, 11.7, 9.97,
            adam=False, opt="SGD", conv_param_frac=0.9, vanilla_slower=True,
            fwd_breaks=0, arch_class="cnn", ratio=0.85,
        ),
        _card(
            "densenet121", "torchbench", "F6", False, -23.0, 321.8, 237.1, 291.6,
            adam=False, opt="SGD", dense_n=58, conv_param_frac=0.4, fwd_breaks=0,
            arch_class="cnn",
        ),
        _card(
            "yolov3", "torchbench", "S1", True, 35.1, 68.8, 68.8, 44.7,
            adam=False, fwd_breaks=25, hot_breaks=True, has_yolo=True, has_setattr=True,
            ratio=1.0, arch_class="cnn",
        ),
        _card(
            "speech_transformer", "torchbench", "S1", True, 5.2, 39.0, 35.4, 33.6,
            adam=True, fwd_graphs=7, fwd_breaks=17, hot_breaks=True, has_speech=True,
            has_nonzero=True, ratio=1.10,
        ),
        _card(
            "detectron2_maskrcnn_r_50_c4", "torchbench", "S4", True, 5.9, 91.3, None, 85.9,
            vanilla_ok=False, conv_param_frac=0.8, fwd_breaks=40, adam=False, opt="SGD",
        ),
        _card(
            "vision_maskrcnn", "torchbench", "identity", False, -4.4, 48.1, 39.6, 41.3,
            vanilla_ok=True, conv_param_frac=0.7, arch_class="cnn",
        ),
        _card(
            "XLNetLMHeadModel", "huggingface", "S3-deep", True, 53.3, 530.0, 291.0, 136.0,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            n_layer=24, n_tensors=411, sync=39, ratio=1.82, model_type="xlnet",
        ),
        _card(
            "ElectraForCausalLM", "huggingface", "F2-skinny", False, 1.6, 79.7, 33.2, 32.7,
            adam=True, hidden=256, n_layer=12, n_params_m=13.5, ratio=2.40, skinny=True,
            zero_break=True, model_type="electra",
        ),
        _card(
            "basic_gnn_gcn", "torchbench", "identity", False, None, 3.85, 3.68, 3.53,
            n_params_m=0.1, vanilla_ms=3.68, arch_class="gnn",
        ),
    ]


def default_extra_trials() -> list:
    """Failed accel probes kept beside the settled (often identity) case card."""
    from agent.memory_tree import trial_from_case

    def fail(case, scheme, vs, eager, vanilla, agent, **fp):
        fp.setdefault("case", case)
        return trial_from_case(
            {
                "case": case,
                "scheme": scheme,
                "beat_naive": False,
                "vs_naive_pct": vs,
                "ok": True,
                "eager_ms": eager,
                "vanilla_ms": vanilla,
                "agent_ms": agent,
                "source": "seed-fail",
                "fp": fp,
            }
        )

    return [
        fail(
            "DistilBertForMaskedLM", "S3-unfused", -68.3, 176.7, 66.7, 112.2,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            n_layer=6, ratio=2.65, model_type="distilbert", skinny=True, sync=64,
            hidden=768, n_params_m=66, arch_class="transformer_base",
        ),
        fail(
            "MegatronBertForCausalLM", "S3-deep", -35.8, 92.2, 57.2, 77.7,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            n_layer=24, ratio=1.61, skinny=True, model_type="megatron-bert",
            hidden=1024, n_params_m=334, arch_class="transformer_base",
        ),
        fail(
            "MobileBertForMaskedLM", "S3-deep", -6.3, 153.3, 87.8, 93.3,
            adam=True, layerdrop_n=0, fwd_graphs=1, fwd_breaks=0, zero_break=True,
            n_layer=24, n_tensors=1117, skinny=True, model_type="mobilebert",
            hidden=128, n_params_m=37,
        ),
        fail(
            "densenet121", "S3-unfused", -23.0, 321.8, 237.1, 291.6,
            adam=False, opt="SGD", dense_n=58, conv_param_frac=0.4, fwd_breaks=0,
            arch_class="cnn",
        ),
    ]


def default_payload() -> dict:
    from agent.memory_tree import DEFAULT_EXPLORE_FROM, trial_from_case

    cases = default_cases()
    return {
        "version": 2,
        "updated": "seed",
        "doc": (
            "Runtime retrieval tree. Hard match = first matching child `when`. "
            "cases = best-so-far per task; trials = append-only win/lose/crash log; "
            "paths = tried-scheme order that later similar tasks can skip. "
            "Similarity is two-channel kNN (wins to copy, losses to ban) on compact "
            "fingerprints — never source or this whole file. After a named scheme is "
            "tried, explore_from lists legal next probes. A large win promotes a tree leaf. "
            "Re-seed with python -m agent.memory_bootstrap. Narrative: memory/scheme_index."
        ),
        "hit_policy": "first_match",
        "schemes": default_schemes(),
        "tree": default_tree(),
        "explore_from": dict(DEFAULT_EXPLORE_FROM),
        "cases": cases,
        "trials": [trial_from_case(c) for c in cases] + default_extra_trials(),
        "paths": [],
        "evolve_log": [],
    }


def write_default(path: Path | None = None) -> Path:
    path = path or TREE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(default_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def main():
    p = write_default()
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
