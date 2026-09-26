"""Official Dynamo *batch* (and a few harness knobs).

Workload for GKO vs official: same model and the official CSV batch.
AMP and optimizer class are not copied from official:

- AMP is a GKO harness flag (HF/TIMM on, TorchBench off). It is a legal
  optimization if correctness passes; gold recipes do not currently toggle it.
- Optimizer *class* is whatever TorchBench / GKO Adam fallback already uses.
  Agent actions only decide whether to compile opt.step or wrap train_step.

compile(train_step) / compile(opt.step) / cudagraphs are methods.
"""

from __future__ import annotations

import csv
from pathlib import Path

GKO = Path(__file__).resolve().parents[1]
DYNAMO_DIR = GKO / "run" / "dynamo_baseline_20260818"

CSV_BY_SUITE = {
    "huggingface": DYNAMO_DIR / "huggingface_training.csv",
    "timm": DYNAMO_DIR / "timm_training.csv",
    "torchbench": DYNAMO_DIR / "torchbench_training_merged.csv",
}

# Dynamo huggingface.yaml only_fp32 (FFT / complex cannot AMP).
# TorchBench yaml fp32_only_models: FakeQuant / PrivacyEngine need fp32.
ONLY_FP32 = frozenset(
    {
        "GoogleFnet",
        "mobilenet_v2_quantized_qat",
        "resnet50_quantized_qat",
        "opacus_cifar10",
    }
)

# torchbench.yaml disable_cudagraph
DISABLE_CUDAGRAPH = frozenset({"tts_angular"})

# PrivacyEngine *is* the workload; replacing it with vanilla Adam is not
# matching official optimizer *kind*, it deletes DP-SGD.
KEEP_BENCH_OPT = frozenset({"opacus_cifar10"})

# TorchBench yaml training overrides (official CSV matches these).
YAML_TRAIN_BATCH = {
    "demucs": 4,
    "dlrm": 1024,
    "densenet121": 4,
    "yolov3": 8,
    "timm_efficientdet": 1,
    "llama_v2_7b_16h": 1,
}

# TorchBench DEFAULT_EVAL_BSIZE for inference-only cases (no train CSV batch).
YAML_EVAL_BATCH = {
    "llava": 1,
    "sam": 32,
}

# Minimal set used by batch/AMP helpers. Full Dynamo mode set lives in
# eval/run_official_dynamo_103.py INFERENCE_ONLY (train-skip / only_inference).
INFERENCE_ONLY = frozenset(
    {
        "llava",
        "sam",
        "M2M100ForConditionalGeneration",
    }
)
DYNAMO_103_DIR = GKO / "run" / "dynamo_103_20260912"

# Cases whose previous GKO numbers cannot be compared to official Dynamo
# (batch mismatch, missing agent row, or official compile-fail).
PRIORITY_REMEASURE = [
    ("torchbench", "densenet121"),
    ("torchbench", "demucs"),
    ("torchbench", "dlrm"),
    ("torchbench", "yolov3"),
    ("torchbench", "pytorch_stargan"),
    ("torchbench", "speech_transformer"),
    ("torchbench", "resnet18"),
    ("torchbench", "detectron2_maskrcnn_r_50_c4"),
    ("torchbench", "mobilenet_v2_quantized_qat"),
    ("torchbench", "resnet50_quantized_qat"),
    ("torchbench", "opacus_cifar10"),
]

# Copied from dynamo/common.py BENCHMARK_USE_SGD so --print-plan works
# without importing torch. Keep in sync if that set changes.
SGD_MODELS = frozenset(
    {
        "BERT_pytorch",
        "LearningToPaint",
        "alexnet",
        "dcgan",
        "demucs",
        "densenet121",
        "dlrm",
        "fastNLP_Bert",
        "mobilenet_v2",
        "phlippe_densenet",
        "phlippe_resnet",
        "pytorch_stargan",
        "resnet18",
        "shufflenet_v2_x1_0",
        "speech_transformer",
        "squeezenet1_1",
        "stable_diffusion_text_encoder",
        "vgg16",
        "AlbertForMaskedLM",
        "BartForCausalLM",
        "ElectraForCausalLM",
        "M2M100ForConditionalGeneration",
        "MBartForCausalLM",
        "OPTForCausalLM",
        "PLBartForCausalLM",
        "PegasusForCausalLM",
        "TrOCRForCausalLM",
        "XGLMForCausalLM",
        "adv_inception_v3",
        "ghostnet_100",
        "tf_efficientnet_b0",
    }
)


def load_official_rows() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for suite, path in CSV_BY_SUITE.items():
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                name = (row.get("name") or "").strip()
                if name:
                    out[(suite, name)] = row
    return out


def is_inference_only(name: str) -> bool:
    base = (name or "").split("/")[-1]
    return name in INFERENCE_ONLY or base in INFERENCE_ONLY


def official_batch(suite: str, name: str, rows: dict | None = None) -> int | None:
    rows = rows if rows is not None else load_official_rows()
    row = rows.get((suite, name))
    if row and row.get("batch_size") not in ("", None):
        bs = int(float(row["batch_size"]))
        if bs > 0:
            return bs
    if is_inference_only(name):
        base = (name or "").split("/")[-1]
        if base in YAML_EVAL_BATCH:
            return int(YAML_EVAL_BATCH[base])
    if suite == "torchbench" and name in YAML_TRAIN_BATCH:
        return int(YAML_TRAIN_BATCH[name])
    return None


def load_dynamo_103_row(suite: str, name: str) -> dict:
    """Prefer the current official 103 campaign (includes --inference llava/sam)."""
    slug = name.replace("/", "__")
    path = DYNAMO_103_DIR / "per_case" / f"{suite}_{slug}.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("name") == name:
                try:
                    if float(row.get("abs_latency") or 0) > 0:
                        return row
                except ValueError:
                    return {}
    return {}


def official_amp(name: str) -> bool:
    base = (name or "").split("/")[-1]
    return name not in ONLY_FP32 and base not in ONLY_FP32


def gko_amp(suite: str, name: str) -> bool:
    """Collector / agent_loop default: AMP on HF and TIMM, off TorchBench."""
    if (suite or "").lower() not in ("huggingface", "hf", "timm"):
        return False
    return official_amp(name)


def official_cudagraphs(name: str) -> bool:
    base = (name or "").split("/")[-1]
    return name not in DISABLE_CUDAGRAPH and base not in DISABLE_CUDAGRAPH


def uses_sgd(name: str) -> bool:
    base = (name or "").split("/")[-1]
    return name in SGD_MODELS or base in SGD_MODELS


def official_opt_name(name: str) -> str:
    if name in KEEP_BENCH_OPT or (name or "").split("/")[-1] in KEEP_BENCH_OPT:
        return "bench"
    return "sgd" if uses_sgd(name) else "adam"


def official_optimizer(name: str, model, *, disable_sgd_step: bool = False):
    """Same class as Dynamo common.BenchmarkRunner.init_optimizer.

    disable_sgd_step=True matches official compile(train_step): SGD.step is
    dynamo-disabled so the compiled step does not include the foreach add.
    Gold recipes that compile_optimizer must *not* disable it.
    """
    import torch

    if name in KEEP_BENCH_OPT or (name or "").split("/")[-1] in KEEP_BENCH_OPT:
        return None
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        params = list(model.parameters())
    if uses_sgd(name):
        opt = torch.optim.SGD(params, lr=0.01, foreach=True)
        if disable_sgd_step:
            opt.step = torch._dynamo.disable(opt.step)
        return opt
    return torch.optim.Adam(params, lr=0.01, capturable=True, foreach=True)


def _frozen_suite_cases(*, skip: set[str] | None = None) -> list[tuple[str, str]]:
    path = GKO / "run" / "gko_eval" / "gko_eval.csv"
    skip = skip or set()
    seen: list[tuple[str, str]] = []
    got: set[tuple[str, str]] = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            suite, model = row.get("suite") or "", row.get("model") or ""
            if suite not in ("huggingface", "timm", "torchbench"):
                continue
            if model in skip:
                continue
            key = (suite, model)
            if key in got:
                continue
            got.add(key)
            seen.append(key)
    return seen


def gko_training_cases() -> list[tuple[str, str]]:
    """Unique HF / TIMM / TorchBench training models from frozen gko_eval.csv."""
    return _frozen_suite_cases(skip={"llava", "sam"})


def all_103_cases() -> list[tuple[str, str]]:
    """Frozen 103 including inference-only llava/sam."""
    cases = _frozen_suite_cases()
    inf = [("torchbench", "llava"), ("torchbench", "sam")]
    rest = [k for k in cases if k not in set(inf)]
    # Inference first so gold-103 can retry after weights land.
    return [k for k in inf if k in set(cases)] + rest


def ordered_cases(
    *, priority_only: bool = False, include_inference: bool = False
) -> list[tuple[str, str]]:
    all_cases = all_103_cases() if include_inference else gko_training_cases()
    got = set(all_cases)
    first = [k for k in PRIORITY_REMEASURE if k in got]
    if include_inference:
        inf = [k for k in (("torchbench", "llava"), ("torchbench", "sam")) if k in got]
        first = inf + [k for k in first if k not in set(inf)]
    if priority_only:
        return first
    rest = [k for k in all_cases if k not in set(first)]
    return first + rest
