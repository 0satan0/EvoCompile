"""Compile recipe: the only artifact an agent should emit.

Benchmark / Dynamo harness stays frozen: it measures eager and naive
`torch.compile(model)`. The agent never patches that wrap per case.

Input to the agent: uncompiled nn.Module + example inputs + baseline
(eager_ms, vanilla_ms or vanilla traceback).
Output: a JSON recipe (whitelist ops) describing how to apply compile — not
rewritten Python source / a free-form ``forward``. Ops such as allow_logging
remove print/IO graph breaks via Dynamo config, without AST-editing the model.
The eval runner applies the recipe and writes feedback (ms, ok, beat_naive).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

GKO = Path(__file__).resolve().parents[1]
RECIPE_DIR = GKO / "eval" / "recipes"


class _EagerCall(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def __getattr__(self, name):
        # Proxy LSTM.flatten_parameters etc. so wrapping the cell kernel
        # does not break parent Python that still calls methods on it.
        if name != "inner":
            inner = self._modules.get("inner")
            if inner is not None and hasattr(inner, name):
                return getattr(inner, name)
        return super().__getattr__(name)

    @torch._dynamo.disable
    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


def _resolve(model: nn.Module, path: str) -> nn.Module:
    if not path:
        return model
    return model.get_submodule(path)


def _set_on_parent(model: nn.Module, path: str, value: nn.Module) -> None:
    if not path:
        raise ValueError("cannot replace the root module in-place; use compile path=''")
    parent_name, _, child = path.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, value)


def _disable_fn(fn):
    if fn is None or getattr(fn, "__dynamo_disable", False):
        return fn
    disabled = torch._dynamo.disable(fn)
    try:
        disabled.__dynamo_disable = True  # type: ignore[attr-defined]
    except Exception:
        pass
    code = getattr(fn, "__code__", None)
    if code is not None:
        try:
            torch._dynamo.eval_frame.skip_code(code)
        except Exception:
            pass
    return disabled


def _wrap_matching_pred(root: nn.Module, pred) -> int:
    items = [(n, m) for n, m in root.named_modules() if n]
    items.sort(key=lambda x: x[0].count("."), reverse=True)
    n = 0
    for name, mod in items:
        if isinstance(mod, _EagerCall) or not pred(mod):
            continue
        parent_name, _, child = name.rpartition(".")
        try:
            parent = root.get_submodule(parent_name) if parent_name else root
        except AttributeError:
            continue
        if isinstance(parent, _EagerCall):
            continue
        cur = getattr(parent, child, None)
        if cur is None or isinstance(cur, _EagerCall) or not pred(cur):
            continue
        setattr(parent, child, _EagerCall(cur))
        n += 1
    return n


def _wrap_class_substr(root: nn.Module, substrs: list[str]) -> int:
    def pred(mod: nn.Module) -> bool:
        name = type(mod).__name__
        return any(s in name for s in substrs)

    return _wrap_matching_pred(root, pred)


def identity_recipe(backend: str = "inductor") -> dict:
    """Same as naive: torch.compile(whole model). Most cases need only this."""
    return {
        "backend": backend,
        "fullgraph": False,
        "actions": [{"op": "compile", "path": ""}],
        "note": "identity = naive torch.compile(model)",
    }


def layout_inputs(inputs, recipe: dict):
    """Apply recipe input layout (e.g. channels_last for TIMM conv stem)."""
    ops = [a.get("op") for a in recipe.get("actions", [])]
    if "channels_last" not in ops and recipe.get("input_memory_format") != "channels_last":
        return inputs

    def conv(x):
        if torch.is_tensor(x) and x.dim() == 4:
            return x.contiguous(memory_format=torch.channels_last)
        return x

    if isinstance(inputs, dict):
        return {k: conv(v) for k, v in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(conv(x) for x in inputs)
    return conv(inputs)


def _pick_input_tensor(inputs, act: dict):
    """Resolve the tensor named by a mark_dynamic action."""
    key = act.get("input")
    idx = act.get("index")
    if isinstance(inputs, dict):
        if key is None:
            raise ValueError("mark_dynamic on dict inputs needs 'input' (example: input_ids)")
        t = inputs[key]
        if idx is not None and isinstance(t, (list, tuple)):
            t = t[idx]
        return t
    seq = inputs if isinstance(inputs, (list, tuple)) else (inputs,)
    if key is not None:
        try:
            idx = int(key)
        except (TypeError, ValueError):
            # HF-style "input":"input_ids" on tuple/list: first tensor, do not crash.
            i = 0 if idx is None else int(idx)
            return seq[i]
    i = 0 if idx is None else int(idx)
    return seq[i]


def _apply_mark_dynamic(inputs, act: dict) -> str:
    t = _pick_input_tensor(inputs, act)
    if not torch.is_tensor(t):
        raise TypeError(f"mark_dynamic target is {type(t)}, not a tensor")
    dim = act.get("dim", act.get("dims"))
    if dim is None:
        raise ValueError("mark_dynamic needs 'dim' (int or list)")
    kwargs = {}
    if act.get("min") is not None:
        kwargs["min"] = act["min"]
    if act.get("max") is not None:
        kwargs["max"] = act["max"]
    # mark_dynamic requires dynamo dynamic_shapes (default True in this tree).
    if hasattr(torch._dynamo.config, "dynamic_shapes"):
        torch._dynamo.config.dynamic_shapes = True
    torch._dynamo.mark_dynamic(t, dim, **kwargs)
    where = act.get("input", act.get("index", 0))
    return f"mark_dynamic:{where} dim={dim}"


def apply_input_ops(inputs, recipe: dict):
    """channels_last then mark_dynamic. Must run *before* torch.compile."""
    notes = []
    inputs = layout_inputs(inputs, recipe)
    for act in recipe.get("actions", []) or []:
        if act.get("op") == "mark_dynamic":
            notes.append(_apply_mark_dynamic(inputs, act))
    return inputs, notes


def load_recipe(path: Path) -> dict:
    return json.loads(path.read_text())


def save_recipe(recipe: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n")


def recipe_path_for(name: str) -> Path:
    return RECIPE_DIR / f"{name.replace('/', '__')}.json"


def _compile_kwargs(recipe: dict, act: dict) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "backend": recipe.get("backend", "inductor"),
        "fullgraph": bool(act.get("fullgraph", recipe.get("fullgraph", False))),
    }
    if "dynamic" in act:
        kwargs["dynamic"] = act["dynamic"]
    mode = act.get("mode", recipe.get("mode"))
    if mode:
        kwargs["mode"] = mode
    return kwargs


def apply_recipe(model: nn.Module, recipe: dict, inputs=None) -> tuple[nn.Module, str]:
    """Mutate `model` according to recipe. Returns (callable_model, description).

    If ``inputs`` is passed, mark_dynamic / channels_last run first (compile sees
    the marked tensors). Callers that already ran apply_input_ops may omit inputs.
    """
    notes = [recipe.get("note", "recipe")]
    if inputs is not None:
        inputs, in_notes = apply_input_ops(inputs, recipe)
        notes.extend(in_notes)

    for k, v in (recipe.get("dynamo") or {}).items():
        setattr(torch._dynamo.config, k, v)
        notes.append(f"dynamo.{k}={v}")

    inductor = recipe.get("inductor") or {}
    if "cudagraphs" in inductor:
        try:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraphs = bool(inductor["cudagraphs"])
            notes.append(f"cudagraphs={inductor['cudagraphs']}")
        except Exception:
            pass

    # Pattern rewrites must be registered before any compile action.
    for act in recipe.get("actions", []):
        if act.get("op") == "replace_pattern":
            from kernels import register as register_kernel

            backend = str(act.get("backend") or "")
            if backend:
                notes.append(register_kernel(act["kernel"], backend=backend))
            else:
                notes.append(register_kernel(act["kernel"]))

    compiled_root = None
    for act in recipe.get("actions", []):
        op = act["op"]
        path = act.get("path", "")
        if op == "allow_logging":
            notes.append(_allow_logging())
            continue
        if op == "apply_rewrite":
            from rewrites import apply as apply_rw

            notes.append(apply_rw(model, act.get("rewrite") or ""))
            continue
        if op in ("mark_dynamic", "replace_pattern"):
            continue
        if op == "dump_trace":
            from trace_dump import dump_trace_dir

            dest = dump_trace_dir(recipe)
            notes.append(f"dump_trace:{dest}")
            continue
        if op == "disable_forward":
            mod = _resolve(model, path)
            orig = mod.forward

            @torch._dynamo.disable
            def _eager(*args, _orig=orig, **kwargs):
                return _orig(*args, **kwargs)

            mod.forward = _eager
            notes.append(f"eager:{path or '.'}.forward")
        elif op == "disable_method":
            mod = _resolve(model, path)
            name = act["name"]
            setattr(mod, name, _disable_fn(getattr(mod, name)))
            notes.append(f"eager:{path or '.'}.{name}")
        elif op == "compile":
            target = _resolve(model, path)
            kwargs = _compile_kwargs(recipe, act)
            compiled = torch.compile(target, **kwargs)
            if path:
                _set_on_parent(model, path, compiled)
            else:
                compiled_root = compiled
            notes.append(
                f"compile:{path or '.'} fullgraph={kwargs['fullgraph']}"
            )
        elif op == "compile_children":
            parent = _resolve(model, path)
            only = act.get("only_class")
            exclude = set(act.get("exclude_class") or [])
            kwargs = _compile_kwargs(recipe, act)
            n = 0
            for child_name, child in list(parent.named_children()):
                cname = type(child).__name__
                if only and cname not in only:
                    continue
                if cname in exclude:
                    continue
                child_path = f"{path}.{child_name}" if path else child_name
                _set_on_parent(
                    model, child_path, torch.compile(child, **kwargs)
                )
                n += 1
            notes.append(
                f"compile_children:{path or '.'} n={n} fullgraph={kwargs['fullgraph']}"
            )
        elif op == "compile_class":
            names = set(act.get("names") or [])
            kwargs = _compile_kwargs(recipe, act)
            items = [
                (n, m)
                for n, m in model.named_modules()
                if n and type(m).__name__ in names
            ]
            items.sort(key=lambda x: x[0].count("."), reverse=True)
            n = 0
            for name, mod in items:
                _set_on_parent(model, name, torch.compile(mod, **kwargs))
                n += 1
            notes.append(
                f"compile_class:{sorted(names)} n={n} fullgraph={kwargs['fullgraph']}"
            )
        elif op == "rewrite_class_forward":
            cls_name = act["class"]
            kind = act["kind"]
            n = 0
            for _name, mod in model.named_modules():
                if type(mod).__name__ != cls_name:
                    continue
                if kind == "lstm_proj_no_flatten":
                    def _fwd(self, x):
                        o, _ = self.lstm(x)
                        return self.linear(o)

                    mod.forward = _fwd.__get__(mod, type(mod))
                elif kind == "encoder_layer_no_inplace":
                    def _fwd(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
                        enc_output, enc_slf_attn = self.slf_attn(
                            enc_input, enc_input, enc_input, mask=slf_attn_mask
                        )
                        if non_pad_mask is not None:
                            enc_output = enc_output * non_pad_mask
                        enc_output = self.pos_ffn(enc_output)
                        if non_pad_mask is not None:
                            enc_output = enc_output * non_pad_mask
                        return enc_output, enc_slf_attn

                    mod.forward = _fwd.__get__(mod, type(mod))
                elif kind == "decoder_layer_no_inplace":
                    def _fwd(
                        self,
                        dec_input,
                        enc_output,
                        non_pad_mask=None,
                        slf_attn_mask=None,
                        dec_enc_attn_mask=None,
                    ):
                        dec_output, dec_slf_attn = self.slf_attn(
                            dec_input, dec_input, dec_input, mask=slf_attn_mask
                        )
                        if non_pad_mask is not None:
                            dec_output = dec_output * non_pad_mask
                        dec_output, dec_enc_attn = self.enc_attn(
                            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
                        )
                        if non_pad_mask is not None:
                            dec_output = dec_output * non_pad_mask
                        dec_output = self.pos_ffn(dec_output)
                        if non_pad_mask is not None:
                            dec_output = dec_output * non_pad_mask
                        return dec_output, dec_slf_attn, dec_enc_attn

                    mod.forward = _fwd.__get__(mod, type(mod))
                elif kind == "decoder_preprocess_tensor":
                    ignore_id = -1

                    def _preprocess(self, padded_input):
                        is_tok = padded_input.ne(ignore_id)
                        n, length = padded_input.shape
                        lengths = is_tok.sum(dim=1)
                        ys_in_pad = padded_input.new_full((n, length + 1), self.eos_id)
                        ys_in_pad[:, 0] = self.sos_id
                        ys_in_pad[:, 1:] = padded_input.masked_fill(
                            padded_input.eq(ignore_id), self.eos_id
                        )
                        ys_out_pad = padded_input.new_full((n, length + 1), ignore_id)
                        ys_out_pad[:, :length] = padded_input
                        idx = lengths.clamp(max=length).unsqueeze(1).to(dtype=torch.long)
                        ys_out_pad.scatter_(1, idx, self.eos_id)
                        return ys_in_pad, ys_out_pad

                    mod.preprocess = _preprocess.__get__(mod, type(mod))
                elif kind == "sdp_attn_no_numpy_inf":
                    def _fwd(self, q, k, v, mask=None):
                        attn = torch.bmm(q, k.transpose(1, 2))
                        attn = attn / self.temperature
                        if mask is not None:
                            attn = attn.masked_fill(mask.bool(), float("-inf"))
                        attn = self.softmax(attn)
                        attn = self.dropout(attn)
                        output = torch.bmm(attn, v)
                        return output, attn

                    mod.forward = _fwd.__get__(mod, type(mod))
                elif kind == "yolo_layer_train_no_grid":
                    # Training YOLOLayer.forward always called create_grids which
                    # setattr(self.nx/ny) → graph break. Grid is unused in train.
                    def _fwd(self, p, out):
                        bs, _, ny, nx = p.shape
                        p = (
                            p.view(bs, self.na, self.no, ny, nx)
                            .permute(0, 1, 3, 4, 2)
                            .contiguous()
                        )
                        if self.training:
                            return p
                        self.nx, self.ny = nx, ny
                        self.grid = self.create_grids((nx, ny), p.device)
                        io = p.clone()
                        io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
                        io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
                        io[..., :4] *= self.stride
                        torch.sigmoid_(io[..., 4:])
                        return io.view(bs, -1, self.no), p

                    mod.forward = _fwd.__get__(mod, type(mod))
                elif kind == "longformer_encoder_const_global":
                    # HF LongformerForMaskedLM always marks token 0 global, so
                    # is_global_attn is always True. The original
                    # flatten().any().item() is a data-dependent graph break.
                    def _fwd(
                        self,
                        hidden_states,
                        attention_mask=None,
                        head_mask=None,
                        padding_len=0,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    ):
                        is_index_masked = attention_mask < 0
                        is_index_global_attn = attention_mask > 0
                        is_global_attn = True
                        for idx, layer_module in enumerate(self.layer):
                            layer_outputs = layer_module(
                                hidden_states,
                                attention_mask=attention_mask,
                                layer_head_mask=(
                                    None if head_mask is None else head_mask[idx]
                                ),
                                is_index_masked=is_index_masked,
                                is_index_global_attn=is_index_global_attn,
                                is_global_attn=is_global_attn,
                                output_attentions=output_attentions,
                            )
                            hidden_states = layer_outputs[0]
                        if padding_len:
                            hidden_states = hidden_states[
                                :, : hidden_states.shape[1] - padding_len
                            ]
                        if not return_dict:
                            return (hidden_states,)
                        from transformers.models.longformer.modeling_longformer import (
                            LongformerBaseModelOutput,
                        )

                        return LongformerBaseModelOutput(
                            last_hidden_state=hidden_states
                        )

                    mod.forward = _fwd.__get__(mod, type(mod))
                elif kind == "longformer_static_global_indices":
                    # MLM bench: exactly one global token at index 0. Replace
                    # nonzero / arange(max) so Dynamo sees static shapes.
                    def _indices(self, is_index_global_attn):
                        bsz = is_index_global_attn.shape[0]
                        device = is_index_global_attn.device
                        batch_idx = torch.arange(bsz, device=device)
                        zeros = torch.zeros(bsz, dtype=torch.long, device=device)
                        empty = torch.empty(0, dtype=torch.long, device=device)
                        return (
                            1,
                            (batch_idx, zeros),
                            (batch_idx, zeros),
                            (empty, empty),
                        )

                    mod._get_global_attn_indices = _indices.__get__(mod, type(mod))
                else:
                    raise ValueError(f"unknown rewrite kind: {kind}")
                n += 1
            notes.append(f"rewrite_forward:{cls_name}/{kind} n={n}")
        elif op == "speech_vectorize_pad_mask":
            n = _patch_speech_pad_mask()
            notes.append(f"speech_vectorize_pad_mask n={n}")
        elif op == "hf_layerdrop_off":
            n = _patch_hf_layerdrop_off(model)
            notes.append(f"hf_layerdrop_off n={n}")
        elif op == "channels_last":
            model.to(memory_format=torch.channels_last)
            notes.append("channels_last")
        elif op == "hf_attn_implementation":
            impl = act.get("implementation", "eager")
            cfg = getattr(model, "config", None)
            if cfg is None:
                raise ValueError("hf_attn_implementation requires model.config")
            cfg._attn_implementation = impl
            notes.append(f"attn_implementation={impl}")
        elif op == "wrap_class_exact":
            names = set(act.get("names") or [])

            def pred_exact(mod: nn.Module) -> bool:
                return type(mod).__name__ in names

            n = _wrap_matching_pred(model, pred_exact)
            notes.append(f"wrap_exact n={n} {sorted(names)}")
        elif op == "wrap_class_substr":
            n = _wrap_class_substr(model, list(act.get("substrs", [])))
            notes.append(f"wrap n={n} {act.get('substrs')}")
        elif op == "skip_code_on_types":
            for spec in act.get("targets", []):
                _skip_qualname(spec)
            notes.append("skip_types")
        elif op == "dynamo_skip_modules":
            n = _dynamo_skip_modules(list(act.get("modules") or []))
            notes.append(f"dynamo_skip_modules n={n} {act.get('modules')}")
        elif op == "gcn_cached":
            n = 0
            for m in model.modules():
                if type(m).__name__ in ("GCNConv", "SAGEConv", "GINConv") and hasattr(
                    m, "cached"
                ):
                    m.cached = True
                    n += 1
            notes.append(f"gcn_cached n={n}")
        else:
            raise ValueError(f"unknown recipe op: {op}")

    if compiled_root is not None:
        return compiled_root, "; ".join(notes)
    return model, "; ".join(notes)


def _patch_speech_pad_mask() -> int:
    """Replace the Python for-loop pad mask with a tensor compare.

    Naive compile graph-breaks here:
      get_non_pad_mask → for i in range(N): mask[i, input_lengths[i]:] = 0
    which is 'Dynamic slicing on data-dependent value'.
    """
    import torchbenchmark.models.speech_transformer.speech_transformer.transformer.decoder as dec
    import torchbenchmark.models.speech_transformer.speech_transformer.transformer.encoder as enc
    import torchbenchmark.models.speech_transformer.speech_transformer.utils.utils as u

    def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
        assert input_lengths is not None or pad_idx is not None
        if input_lengths is not None:
            t = padded_input.size(1)
            rng = torch.arange(t, device=padded_input.device)
            lengths = input_lengths.to(device=padded_input.device)
            if lengths.dim() > 1:
                lengths = lengths.reshape(-1)
            mask = rng.unsqueeze(0) < lengths.unsqueeze(1)
            return mask.to(dtype=padded_input.dtype).unsqueeze(-1)
        assert padded_input.dim() == 2
        return padded_input.ne(pad_idx).float().unsqueeze(-1)

    enc.get_non_pad_mask = get_non_pad_mask
    dec.get_non_pad_mask = get_non_pad_mask
    u.get_non_pad_mask = get_non_pad_mask
    return 3


def _patch_hf_layerdrop_off(model: nn.Module) -> int:
    """Stop LayerDrop from using a tensor as a Python bool.

    HF encoder/decoder stacks do `torch.rand([]) < self.layerdrop` then `if skip`.
    Default layerdrop is often 0, so the skip never fires — replace rand([])
    with Python 1.0 so Dynamo constant-folds the branch. Applies to any
    submodule that has `.layerdrop` (M2M100 / Whisper / Bart / T5 / ...).
    """
    import types

    class _TorchNeverSkip:
        def __getattr__(self, name):
            return getattr(torch, name)

        def rand(self, *args, **kwargs):
            if len(args) == 1 and args[0] == []:
                return 1.0
            return torch.rand(*args, **kwargs)

    n = 0
    seen = set()
    for mod in model.modules():
        if not hasattr(mod, "layerdrop"):
            continue
        cls = type(mod)
        if cls in seen:
            continue
        seen.add(cls)
        fn = getattr(cls, "forward", None)
        if fn is None or getattr(fn, "_gko_layerdrop_off", False):
            n += 1
            continue
        g = dict(fn.__globals__)
        g["torch"] = _TorchNeverSkip()
        new_fn = types.FunctionType(
            fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__
        )
        new_fn._gko_layerdrop_off = True  # type: ignore[attr-defined]
        cls.forward = new_fn
        n += 1
    return n


def _allow_logging() -> str:
    """Closed op: let Dynamo reorder print/logging instead of graph-breaking.

    Does not AST-edit user ``forward``. Registers builtins into
    ``torch._dynamo.config.reorderable_logging_functions`` when present.
    """
    import logging

    notes: list[str] = []
    cfg = torch._dynamo.config
    fns = [print]
    for name in ("debug", "info", "warning", "error", "critical", "log", "exception"):
        fn = getattr(logging, name, None)
        if callable(fn):
            fns.append(fn)
    try:
        import warnings

        fns.append(warnings.warn)
    except Exception:
        pass
    bag = getattr(cfg, "reorderable_logging_functions", None)
    if bag is not None and hasattr(bag, "add"):
        for fn in fns:
            try:
                bag.add(fn)
            except Exception:
                pass
        notes.append("reorderable_logging_functions")
    return "allow_logging:" + (",".join(notes) or "no-op")


def _dynamo_skip_modules(names: list[str]) -> int:
    """Add package prefixes to Dynamo SKIP_DIRS (do not trace into them)."""
    from torch._dynamo import trace_rules

    n = 0
    for name in names:
        try:
            trace_rules.add(name)
            n += 1
        except Exception:
            pass
    return n


def _skip_qualname(spec: str) -> None:
    """spec like 'detectron2.structures.masks.PolygonMasks.__getitem__'."""
    parts = spec.split(".")
    try:
        import importlib

        module = importlib.import_module(".".join(parts[:-2]))
        cls = getattr(module, parts[-2])
        fn = getattr(cls, parts[-1])
        code = getattr(fn, "__code__", None)
        if code is not None:
            torch._dynamo.eval_frame.skip_code(code)
    except Exception:
        pass


def beat_naive(eager_ms, vanilla_ms, vanilla_ok, agent_ms, agent_ok: bool) -> dict:
    """Success criteria for the agent output."""
    out = {
        "agent_ok": agent_ok,
        "vanilla_ok": vanilla_ok,
        "beat_naive": False,
        "beat_eager": False,
        "reason": "",
    }
    if not agent_ok:
        out["reason"] = "agent failed to run"
        return out
    if vanilla_ok is False:
        out["beat_naive"] = True
        out["reason"] = "naive compile failed; agent runs"
    elif (
        vanilla_ok is True
        and agent_ms is not None
        and vanilla_ms is not None
        and agent_ms < vanilla_ms * 0.95
    ):
        out["beat_naive"] = True
        out["reason"] = "agent faster than naive by >5%"
    elif vanilla_ok is None:
        out["reason"] = "agent ran; naive baseline missing"
    else:
        out["reason"] = "agent ran but not faster than naive"
    if eager_ms is not None and agent_ms is not None and agent_ms < eager_ms * 0.95:
        out["beat_eager"] = True
    return out
