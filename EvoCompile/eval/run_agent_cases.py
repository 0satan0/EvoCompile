#!/usr/bin/env python3
"""Eager vs vanilla torch.compile(model) vs agent compile-boundary.

Training cases time fwd+loss+bwd+opt. Inference-only cases (llava/sam)
have no train loop: time no_grad forward_pass, matching Dynamo --inference.

QAT / Detectron / Opacus: eager works, naive compile(model) often does not
→ agent task (region compile). Compiling the Python train_step instead can
graph-break around those modules and hide the incompatibility.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
import traceback
from pathlib import Path

GKO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GKO / "eval"))
sys.path.insert(0, str(GKO / "pytorch" / "benchmarks" / "dynamo"))
sys.path.insert(0, str(GKO / "benchmark"))

from schema import FIELDS, empty_row  # noqa: E402
from agent_compile import apply_agent  # noqa: E402

CASES = [
    "mobilenet_v2_quantized_qat",
    "resnet50_quantized_qat",
    "opacus_cifar10",
    "detectron2_maskrcnn_r_50_c4",
]

# Keep in sync with eval/run_official_dynamo_103.py INFERENCE_ONLY.
# TorchBench: torchbench.yaml skip.test.training. HF: only_inference.
INFERENCE_ONLY = frozenset(
    {
        "llava",
        "sam",
        "M2M100ForConditionalGeneration",
        "cm3leon_generate",
        "detectron2_fasterrcnn_r_101_c4",
        "detectron2_fasterrcnn_r_101_dc5",
        "detectron2_fasterrcnn_r_101_fpn",
        "detectron2_fasterrcnn_r_50_c4",
        "detectron2_fasterrcnn_r_50_dc5",
        "detectron2_fasterrcnn_r_50_fpn",
        "detectron2_fcos_r_50_fpn",
        "detectron2_maskrcnn_r_101_c4",
        "detectron2_maskrcnn_r_101_fpn",
        "detectron2_maskrcnn_r_50_fpn",
        "doctr_det_predictor",
        "doctr_reco_predictor",
        "llama",
        "llama_v2_7b_16h",
        "maml",
        "moondream",
        "pyhpc_equation_of_state",
        "pyhpc_isoneutral_mixing",
        "pyhpc_turbulent_kinetic_energy",
        "sam_fast",
        "simple_gpt",
    }
)


def is_inference_only(name: str) -> bool:
    base = (name or "").split("/")[-1]
    return name in INFERENCE_ONLY or base in INFERENCE_ONLY


def _setup_cwd():
    bench = GKO / "benchmark"
    os.chdir(bench)
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))


def _ensure_dist(device: str = "cuda") -> None:
    import os
    import torch
    import torch.distributed as dist

    torch.backends.cuda.matmul.allow_tf32 = True
    if dist.is_initialized():
        return
    # Single-node: force loopback. torchrun --standalone often sets MASTER_ADDR to the
    # hostname, which is unreachable on some clusters (c10d connect timeout).
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ or not os.environ["MASTER_PORT"]:
        os.environ["MASTER_PORT"] = os.environ.get("GKO_MASTER_PORT", "29777")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", os.environ.get("RANK", "0"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank % max(torch.cuda.device_count(), 1))
    port = os.environ["MASTER_PORT"]
    try:
        dist.init_process_group(
            backend="nccl" if device == "cuda" else "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
    except Exception as e:
        print(f"dist init skipped: {e}", flush=True)


def _adam(model):
    import torch

    return torch.optim.Adam(model.parameters(), lr=0.01)


def _load_torchbench(name: str, device: str = "cuda", batch_size=None):
    import importlib

    # Py3.8: some deps access importlib.resources without importing the submodule.
    import importlib.resources  # noqa: F401

    try:
        import importlib_resources  # noqa: F401
    except Exception:
        pass

    candidates = [
        f"torchbenchmark.models.{name}",
        f"torchbenchmark.canary_models.{name}",
    ]
    errors = []
    mod = None
    for c in candidates:
        try:
            mod = importlib.import_module(c)
            break
        except Exception as e:
            errors.append(f"{c}: {type(e).__name__}: {e}")
    if mod is None:
        raise ImportError(" | ".join(errors))
    test = "eval" if is_inference_only(name) else "train"
    kwargs = {"test": test, "device": device}
    if batch_size is not None:
        kwargs["batch_size"] = int(batch_size)
    # simple_gpt DTensor path requires _world_size == visible CUDA device count.
    # Prefer torchrun WORLD_SIZE when multiprocess; else fall back to device_count.
    if name == "simple_gpt":
        import os

        import torch

        _ensure_dist(device)
        ws = int(os.environ.get("WORLD_SIZE", "0")) or max(torch.cuda.device_count(), 1)
        rank = int(os.environ.get("RANK", "0"))
        kwargs["extra_args"] = [
            "--world_size",
            str(ws),
            "--rank",
            str(rank),
        ]
    bm = mod.Model(**kwargs)
    model, example_inputs = bm.get_module()
    if test == "eval":
        model.eval()
    else:
        model.train()
    # llama_v2_7b_16h from_config is fp32; FlashAttention kernels need fp16/bf16.
    if name == "llama_v2_7b_16h":
        import torch

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
        else:
            model = model.half()
    # doctr get_module omits return_model_output=True; wrap so outputs contain tensors.
    if name in ("doctr_det_predictor", "doctr_reco_predictor"):
        import torch.nn as nn

        inner = model

        class _DoctrWrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m(x, return_model_output=True)
                if isinstance(out, dict) and "out_map" in out:
                    return out["out_map"]
                return out

        model = _DoctrWrap(inner)
    # Dynamo torchbench.py overrides yolov3 train inputs: get_module()
    # returns the dataloader, which is not Darknet.forward(*args).
    if name == "yolov3":
        import torch

        bs = int(getattr(bm, "batch_size", 16) or 16)
        example_inputs = (torch.rand(bs, 3, 384, 512, device=device),)
    elif name == "vision_maskrcnn":
        # get_module() is eval-only: (images,) and model.eval(), because
        # Dynamo compares top-4 boxes. Real train is model(images, targets).
        images, targets = bm.data_loader[0]
        example_inputs = (images, targets)
        model.train()
        print(
            f"vision_maskrcnn train inputs: n_images={len(images)} "
            f"target_keys={list(targets[0]) if targets else []}",
            flush=True,
        )
    def _is_opt(obj):
        return obj is not None and callable(getattr(obj, "zero_grad", None)) and callable(
            getattr(obj, "step", None)
        )

    # BenchmarkModel.get_optimizer() returns self.opt, which some models use
    # as an argparse Namespace (Background_Matting), not torch.optim.
    opt = getattr(bm, "optimizer", None)
    if not _is_opt(opt):
        try:
            opt = bm.get_optimizer()
        except Exception:
            opt = None
    if not _is_opt(opt):
        opt = getattr(bm, "optimizerG", None)
    if is_inference_only(name):
        return bm, model, example_inputs, None
    if not _is_opt(opt):
        opt = _adam(model)
    return bm, model, example_inputs, opt


def _load_huggingface(name: str, device: str = "cuda", batch_size=None):
    """Same construction as dynamo/huggingface.py HuggingfaceRunner.load_model."""
    from common import load_yaml_file
    from huggingface import (
        BATCH_SIZE_KNOWN_MODELS,
        EXTRA_MODELS,
        _ms_pretrained_id,
        generate_inputs_for_model,
        get_module_cls_by_model_name,
    )
    from huggingface_llm_models import HF_LLM_MODELS
    from transformers import AutoConfig

    if batch_size is None:
        divisors = load_yaml_file("huggingface.yaml").get("batch_size", {}).get(
            "divisors", {}
        )
        batch_size = BATCH_SIZE_KNOWN_MODELS.get(name, 16)
        if name in divisors:
            batch_size = max(int(batch_size / divisors[name]), 1)
    else:
        batch_size = int(batch_size)

    if name in HF_LLM_MODELS:
        model, example_inputs = HF_LLM_MODELS[name].get_train_model_and_inputs(
            name, device, batch_size
        )
        model.train()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        print(
            f"loaded huggingface llm {name} batch_size={batch_size} "
            f"input_keys={list(example_inputs)}",
            flush=True,
        )
        return None, model, example_inputs, _adam(model)

    # Match dynamo/huggingface.py HuggingfaceRunner._get_model_cls_and_config:
    # EXTRA_MODELS read ModelScope local snapshots; some classes need pad_token
    # for batch_size > 1. Do not download from hub when offline.
    if name not in EXTRA_MODELS:
        model_cls = get_module_cls_by_model_name(name)
        config = model_cls.config_class()
        if (
            model_cls.__name__
            in (
                "GPT2ForSequenceClassification",
                "GPTNeoForSequenceClassification",
                "GPTJForSequenceClassification",
            )
            or model_cls.__name__.startswith("Roberta")
            or model_cls.__name__.startswith("Marian")
        ):
            config.pad_token_id = 0
    else:
        repo, model_cls = EXTRA_MODELS[name]
        if repo is None:
            from transformers import ReformerConfig

            config = ReformerConfig()
        else:
            config = AutoConfig.from_pretrained(_ms_pretrained_id(repo))
    if "auto" in model_cls.__module__:
        model = model_cls.from_config(config)
    else:
        model = model_cls(config)
    model = model.to(device)
    model.train()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    example_inputs = generate_inputs_for_model(
        model_cls, model, name, batch_size, device, include_loss_args=True
    )
    print(
        f"loaded huggingface {name} batch_size={batch_size} "
        f"input_keys={list(example_inputs)}",
        flush=True,
    )
    return None, model, example_inputs, _adam(model)


def _load_timm(name: str, device: str = "cuda", batch_size=None):
    """Same construction as dynamo/timm_models.py TimmRunner.load_model."""
    import torch
    from common import load_yaml_file
    from timm.data import resolve_data_config
    from timm_models import TIMM_MODELS, TimmRunner

    runner = TimmRunner()
    runner._args = type("A", (), {"channels_last": False, "training": True})()
    model = runner._download_model(name)
    if model is None:
        raise RuntimeError(f"Failed to load TIMM model {name}")
    model.to(device)
    model.train()
    data_config = resolve_data_config({}, model=model, use_test_size=False)
    input_size = data_config["input_size"]
    if batch_size is None:
        batch_size = TIMM_MODELS[name]
        divisors = load_yaml_file("timm_models.yaml").get("batch_size", {}).get(
            "divisors", {}
        )
        if name in divisors:
            batch_size = max(int(batch_size / divisors[name]), 1)
    else:
        batch_size = int(batch_size)
    torch.manual_seed(1337)
    raw = torch.randint(256, size=(batch_size,) + input_size, device=device).to(
        dtype=torch.float32
    )
    example_inputs = [(raw - raw.mean()) / raw.std()]
    print(
        f"loaded timm {name} batch_size={batch_size} input_size={input_size}",
        flush=True,
    )
    return None, model, example_inputs, _adam(model)


def _detect_suite(name: str, suite: str = "") -> str:
    suite = (suite or "").lower()
    if suite:
        return "huggingface" if suite in ("huggingface", "hf") else suite
    dynamo_dir = str(GKO / "pytorch" / "benchmarks" / "dynamo")
    if dynamo_dir not in sys.path:
        sys.path.insert(0, dynamo_dir)
    try:
        from huggingface import BATCH_SIZE_KNOWN_MODELS, EXTRA_MODELS

        if name in BATCH_SIZE_KNOWN_MODELS or name in EXTRA_MODELS:
            return "huggingface"
    except Exception:
        pass
    try:
        from timm_models import TIMM_MODELS

        if name in TIMM_MODELS:
            return "timm"
    except Exception:
        pass
    return "torchbench"


def _load(name: str, device: str = "cuda", suite: str = "", batch_size=None):
    _ensure_dist(device)
    dynamo_dir = str(GKO / "pytorch" / "benchmarks" / "dynamo")
    if dynamo_dir not in sys.path:
        sys.path.insert(0, dynamo_dir)
    suite = _detect_suite(name, suite)
    # Default to official Dynamo CSV batch (suite119 standard) when not overridden.
    if batch_size is None:
        try:
            from official_workload import official_batch

            batch_size = official_batch(suite, name)
        except Exception:
            batch_size = None
    print(f"suite={suite} case={name} batch_size={batch_size}", flush=True)
    if suite == "huggingface":
        return _load_huggingface(name, device, batch_size=batch_size)
    if suite == "timm":
        return _load_timm(name, device, batch_size=batch_size)
    return _load_torchbench(name, device, batch_size=batch_size)


def _opt_zero(opt):
    try:
        opt.zero_grad(set_to_none=True)
    except TypeError:
        opt.zero_grad()


def _loss_to_float(loss):
    import torch

    if not torch.is_tensor(loss):
        raise RuntimeError("loss is not a tensor")
    t = loss.detach()
    return float(t.item()) if t.numel() == 1 else float(t.sum().item())


def _compute_loss(pred):
    from torch._dynamo.testing import reduce_to_scalar_loss

    if hasattr(pred, "loss") and pred.loss is not None:
        return pred.loss
    return reduce_to_scalar_loss(pred)


def _reduce_pred(pred):
    """Scalar stand-in for inference outputs (allclose vs eager)."""
    import torch

    tensors = []

    def walk(x):
        if torch.is_tensor(x):
            tensors.append(x.detach().float().reshape(-1))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif hasattr(x, "get_fields") and callable(x.get_fields):
            # detectron2.structures.Instances
            try:
                walk(x.get_fields())
            except Exception:
                pass
        elif hasattr(x, "tensor") and torch.is_tensor(getattr(x, "tensor", None)):
            # detectron2.structures.Boxes / BitMasks
            walk(x.tensor)
        elif hasattr(x, "logits"):
            walk(x.logits)
        elif hasattr(x, "loss") and x.loss is not None:
            walk(x.loss)

    walk(pred)
    if not tensors:
        raise RuntimeError("no tensor outputs to reduce")
    return float(torch.cat(tensors).sum().item())


def _eval_fn(model, inputs, opt=None, amp: bool = False, *, allow_grad: bool = False):
    """Forward for inference-only cases. Signature matches _step_fn.

    allow_grad: maml (and similar) run an inner autograd loop inside forward;
    torch.no_grad() would raise 'does not require grad'.
    """
    import torch

    ctx = torch.cuda.amp.autocast(enabled=bool(amp) and torch.cuda.is_available())
    grad_ctx = torch.enable_grad() if allow_grad else torch.no_grad()
    with grad_ctx, ctx:
        if isinstance(inputs, dict):
            pred = model(**inputs)
        else:
            pred = model(*inputs)
    return _reduce_pred(pred)


def runtime_step(name: str):
    """Train step, or inference forward_pass for llava/sam."""
    if is_inference_only(name):
        # maml: meta inner-loop needs grad under official --inference.
        if name == "maml" or name.startswith("maml"):

            def _maml_eval(model, inputs, opt=None, amp: bool = False):
                return _eval_fn(model, inputs, opt, amp=amp, allow_grad=True)

            return _maml_eval
        return _eval_fn
    return _step_fn


def _fwd_bwd(model, inputs, amp: bool = False):
    """Forward + scalar loss + backward. Optimizer stays outside this function.

    Compiling this (not opt.zero_grad / opt.step) is the usual production wrap:
    same as Dynamo's train-step compile with optimizer carved out.
    """
    import torch
    from torch._dynamo.utils import clone_inputs

    cloned = clone_inputs(inputs)
    ctx = torch.cuda.amp.autocast(enabled=bool(amp) and torch.cuda.is_available())
    with ctx:
        if isinstance(cloned, dict):
            pred = model(**cloned)
        else:
            pred = model(*cloned)
        loss = _compute_loss(pred)
    if not torch.is_tensor(loss):
        raise RuntimeError("loss is not a tensor")
    loss.backward()
    return loss


def _step_fn(model, inputs, opt, amp: bool = False):
    """Full train step, including optimizer (Dynamo harness shape)."""
    _opt_zero(opt)
    loss = _fwd_bwd(model, inputs, amp=amp)
    opt.step()
    return _loss_to_float(loss)


def _time_loop(fn, *, warmup: int, measure: int, device: str = "cuda"):
    import torch

    last_loss = None
    t_compile0 = time.perf_counter()
    for i in range(warmup):
        last_loss = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        if i == 0:
            compile_s = time.perf_counter() - t_compile0
    if warmup == 0:
        compile_s = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(measure):
        last_loss = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / max(measure, 1) * 1000.0
    peak = 0.0
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024**3)
    return ms, compile_s, peak, last_loss


def _run_variant(name, model, inputs, opt, variant, warmup, measure):
    import torch

    body = runtime_step(name)

    def raw():
        return body(model, inputs, opt)

    if variant == "eager":
        return _time_loop(raw, warmup=warmup, measure=measure)
    if variant == "vanilla":
        # Naive wrap of the nn.Module — same class of failure as Dynamo harness
        # (FakeQuant size-assert / PolygonMasks / PrivacyEngine hooks).
        # Compiling the Python train_step instead allows graph breaks and can
        # hide the real compile-compat problem.
        cmodel = torch.compile(model, backend="inductor", fullgraph=False)

        def vanilla_step():
            return body(cmodel, inputs, opt)

        return _time_loop(vanilla_step, warmup=max(warmup, 3), measure=measure)
    if variant == "agent":
        model2, desc = apply_agent(name, model)
        print(f"  [{name}] {desc}", flush=True)

        def agent_step():
            return body(model2, inputs, opt)

        return _time_loop(agent_step, warmup=max(warmup, 3), measure=measure)
    raise ValueError(variant)


def run_case(name: str, args) -> dict:
    import torch

    row = empty_row(
        case_id=f"agent_{name}",
        suite="torchbench",
        model=name,
        task="training",
        gpu_count="1",
        parallelism="none",
        compile_config="",
        status="error",
    )
    results = {}
    variants = tuple(args.variants)
    for variant in variants:
        print(f"== {name} {variant} ==", flush=True)
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        gc.collect()
        try:
            bm, model, inputs, opt = _load(name)
            ms, compile_s, peak, loss = _run_variant(
                name, model, inputs, opt, variant, args.warmup, args.measure
            )
            results[variant] = {
                "ms": ms,
                "compile_s": compile_s,
                "peak": peak,
                "loss": loss,
                "ok": True,
                "err": "",
            }
            print(
                f"  {variant}: {ms:.2f} ms  compile={compile_s:.1f}s  "
                f"peak={peak:.2f}GiB  loss={loss}",
                flush=True,
            )
        except Exception:
            err = traceback.format_exc()
            print(err, flush=True)
            results[variant] = {
                "ms": "",
                "compile_s": "",
                "peak": "",
                "loss": "",
                "ok": False,
                "err": err.splitlines()[-1][:200],
            }
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    missing = {"ok": False, "ms": "", "compile_s": "", "peak": "", "loss": "", "err": "skipped"}
    eager = results.get("eager", missing)
    vanilla = results.get("vanilla", missing)
    agent = results.get("agent", missing)
    vanilla_ran = "vanilla" in results
    agent_ran = "agent" in results
    row["eager_or_uncompiled_step_ms"] = (
        f"{eager['ms']:.4f}" if eager["ok"] else ""
    )
    row["vanilla_compile_step_ms"] = (
        f"{vanilla['ms']:.4f}" if vanilla["ok"] else ""
    )
    row["agent_step_ms"] = f"{agent['ms']:.4f}" if agent["ok"] else ""
    row["compile_time"] = (
        f"{agent['compile_s']:.4f}"
        if agent["ok"]
        else (f"{vanilla['compile_s']:.4f}" if vanilla["ok"] else "")
    )
    row["peak_memory"] = (
        f"{agent['peak']:.4f}"
        if agent["ok"]
        else (f"{vanilla['peak']:.4f}" if vanilla["ok"] else "")
    )
    if eager["ok"] and vanilla_ran and not vanilla["ok"] and agent["ok"]:
        row["compile_config"] = "agent_compile_boundary"
        row["status"] = "pass:agent"
        row["correctness"] = "n/a"
    elif eager["ok"] and vanilla["ok"]:
        row["compile_config"] = "naive_inductor"
        row["status"] = "pass"
        if agent["ok"]:
            row["compile_config"] = "naive_or_agent"
            ve, ae = vanilla["ms"], agent["ms"]
            if ae < ve * 0.95:
                row["status"] = "pass:agent_faster"
            elif ae > eager["ms"] * 1.05 and ve > eager["ms"] * 1.05:
                row["status"] = "pass:compile_no_speedup"
    elif eager["ok"] and vanilla_ran and not vanilla["ok"] and agent_ran and not agent["ok"]:
        row["compile_config"] = "vanilla_fail; agent_fail"
        row["status"] = f"fail:vanilla={vanilla['err']}; agent={agent['err']}"
        row["correctness"] = "n/a"
    elif eager["ok"] and agent["ok"] and not vanilla_ran:
        row["compile_config"] = "agent_compile_boundary"
        row["status"] = "pass:agent"
    elif not eager["ok"] and "eager" in results:
        row["status"] = f"fail:eager={eager['err']}"
    else:
        row["status"] = "partial"
    return row


def _upsert(path: Path, row: dict):
    rows = []
    if path.exists():
        with path.open() as f:
            rows = list(csv.DictReader(f))
    by = {r.get("case_id"): r for r in rows}
    prev = by.get(row["case_id"], {})
    merged = {}
    for k in FIELDS:
        newv = row.get(k, "")
        oldv = prev.get(k, "")
        merged[k] = newv if newv not in ("", None) else oldv
    if row.get("status", "").startswith("pass"):
        merged["status"] = row["status"]
        if row.get("compile_config"):
            merged["compile_config"] = row["compile_config"]
    elif row.get("status") in ("partial", "error") and prev.get("status"):
        merged["status"] = prev["status"]
    by[row["case_id"]] = merged
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in by:
            w.writerow({kk: by[k].get(kk, "") for kk in FIELDS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=",".join(CASES))
    p.add_argument(
        "--variants",
        default="eager,vanilla,agent",
        help="Comma-separated: eager,vanilla,agent",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--measure", type=int, default=20)
    p.add_argument(
        "--csv",
        default=str(GKO / "run" / "gko_eval" / "gko_eval.csv"),
    )
    args = p.parse_args()
    args.variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    _setup_cwd()
    names = [x.strip() for x in args.only.split(",") if x.strip()]
    csv_path = Path(args.csv)
    for name in names:
        row = run_case(name, args)
        _upsert(csv_path, row)
        print(f"wrote {row['case_id']} status={row['status']}", flush=True)


if __name__ == "__main__":
    main()
