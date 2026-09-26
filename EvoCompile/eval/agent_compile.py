"""Agent-style compile-boundary helpers.

These cases eager-run; naive torch.compile of the nn.Module typically does not.
Keep Python / data-dependent / FakeQuant / PrivacyEngine hooks eager, and only
compile tensor-heavy regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from recipe import apply_recipe, identity_recipe, load_recipe, recipe_path_for


class _EagerCall(nn.Module):
    """Submodule Dynamo cannot inline: always a graph-break into eager."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    @torch._dynamo.disable
    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


def _swap_matching(root: nn.Module, pred) -> int:
    # Deepest names first so nested FakeQuant/observers are wrapped before
    # their parent. After a parent is replaced with _EagerCall, skip leftovers.
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


def _is_fakequant_or_observer(mod: nn.Module) -> bool:
    # Wrap FakeQuantize only. Nested Observers stay in place so
    # FakeQuantize.forward can still read activation_post_process.min_val.
    name = type(mod).__name__
    return "FakeQuantize" in name or name.endswith("FakeQuant")


def _disable_fn(fn):
    if fn is None or getattr(fn, "__dynamo_disable", False):
        return fn
    disabled = torch._dynamo.disable(fn)
    try:
        disabled.__dynamo_disable = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch._dynamo.eval_frame.skip_code(fn.__code__)
    except Exception:
        pass
    return disabled


def apply_qat_boundary(model: nn.Module, backend: str = "inductor") -> str:
    """FakeQuant / observers stay eager; compile the remaining GraphModule.

    Naive compile inlines fused_moving_avg_obs_fake_quant and hits FakeTensor /
    assert_size_stride. Wrapping those modules forces a compile boundary.
    """
    try:
        torch._dynamo.eval_frame.skip_code(_EagerCall.forward.__code__)
    except Exception:
        pass
    n_skip = _swap_matching(model, _is_fakequant_or_observer)
    compiled = torch.compile(model, backend=backend, fullgraph=False)
    model._gko_compiled = compiled  # type: ignore[attr-defined]
    return (
        f"agent:FakeQuant/observer eager-wrap n={n_skip}; "
        f"compile rest backend={backend} fullgraph=False"
    )


def apply_opacus_boundary(model: nn.Module, backend: str = "inductor") -> str:
    """Compile inner ResNet; keep PrivacyEngine GradSample hooks in eager.

    Does not change Opacus math or disable inductor size asserts.
    """
    inner = getattr(model, "_module", None)
    if inner is None:
        raise TypeError(f"expected GradSampleModule, got {type(model).__name__}")
    model.forward = _disable_fn(model.forward)
    model._module = torch.compile(inner, backend=backend, fullgraph=False)
    for hook_name in ("capture_activations_hook", "capture_backprops_hook"):
        if hasattr(model, hook_name):
            setattr(model, hook_name, _disable_fn(getattr(model, hook_name)))
    return (
        "agent:compile inner ResNet; GradSampleModule/hooks eager "
        f"backend={backend}"
    )


def _skip_detectron2_python():
    """PolygonMasks / Instances / Boxes are Dynamo poison; never trace them."""
    try:
        from detectron2.structures.boxes import Boxes, pairwise_iou
        from detectron2.structures.instances import Instances
        from detectron2.structures.masks import PolygonMasks
    except Exception:
        return
    for obj in (
        getattr(PolygonMasks, "__getitem__", None),
        getattr(PolygonMasks, "crop_and_resize", None),
        getattr(PolygonMasks, "area", None),
        getattr(PolygonMasks, "nonempty", None),
        getattr(Instances, "__getitem__", None),
        getattr(Instances, "to", None),
        getattr(Instances, "get", None),
        getattr(Boxes, "__getitem__", None),
        getattr(Boxes, "to", None),
        pairwise_iou,
    ):
        if obj is None:
            continue
        code = getattr(obj, "__code__", None)
        if code is not None:
            try:
                torch._dynamo.eval_frame.skip_code(code)
            except Exception:
                pass


def apply_detectron2_boundary(model: nn.Module, backend: str = "inductor") -> str:
    """Backbone + ROI tensor heads compiled; RPN / Polygon / sampling eager.

    torch.compile installs a global eval_frame hook, so compiling only the
    backbone still traces PolygonMasks unless the meta-arch stays disabled.
    """
    try:
        import torch._inductor.config as inductor_config

        inductor_config.triton.cudagraphs = False
    except Exception:
        pass
    _skip_detectron2_python()

    orig_fwd = model.forward

    @torch._dynamo.disable
    def _eager_meta(*args, **kwargs):
        return orig_fwd(*args, **kwargs)

    model.forward = _eager_meta

    compiled = []
    if hasattr(model, "backbone"):
        model.backbone = torch.compile(
            model.backbone, backend=backend, fullgraph=False, dynamic=False
        )
        compiled.append("backbone")
    if hasattr(model, "proposal_generator") and model.proposal_generator is not None:
        model.proposal_generator.forward = _disable_fn(model.proposal_generator.forward)
        compiled.append("RPN=eager")
    roi = getattr(model, "roi_heads", None)
    if roi is not None:
        # Keep all ROI Python (PolygonMasks, sampling, pooler) eager.
        # Compiling res5/box/mask with dynamic RoI counts ran (110ms) but
        # was slower than eager (92ms); backbone is the stable tensor region.
        roi.forward = _disable_fn(roi.forward)
        if hasattr(roi, "label_and_sample_proposals"):
            roi.label_and_sample_proposals = _disable_fn(roi.label_and_sample_proposals)
        compiled.append("roi=eager")
    return "agent: " + ", ".join(compiled) + f" backend={backend} cudagraphs=off"


STRATEGIES = {
    "mobilenet_v2_quantized_qat": apply_qat_boundary,
    "resnet50_quantized_qat": apply_qat_boundary,
    "opacus_cifar10": apply_opacus_boundary,
    "detectron2_maskrcnn_r_50_c4": apply_detectron2_boundary,
}


def apply_agent(name: str, model: nn.Module, backend: str = "inductor"):
    path = recipe_path_for(name)
    if path.exists():
        recipe = load_recipe(path)
        recipe.setdefault("backend", backend)
        return apply_recipe(model, recipe)
    if name in STRATEGIES:
        desc = STRATEGIES[name](model, backend=backend)
        compiled = getattr(model, "_gko_compiled", None)
        if compiled is not None:
            return compiled, desc
        return model, desc
    return apply_recipe(model, identity_recipe(backend))
