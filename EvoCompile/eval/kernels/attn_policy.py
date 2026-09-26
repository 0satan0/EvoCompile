"""Feature-based Triton vs CUTLASS dispatch for the 28-case overlay.

Resolved once at apply() (Dynamo-safe: fused_attention only reads an env
string). Features, not a model-name whitelist:

Triton
  - 4D relative / window-rel bias leftover (t5_rel, xlnet, swin_window):
    tiled FA reads broadcast strides; packing SxS for CUTLASS is the tax.
  - logit-softcap (gemma_rope): tanh cap is a Triton epilogue, not CUTLASS GEMM.
  - small-S dense pad / shallow stacks: SxS lives in L2, leftover cuBLAS bmm
    plus a fused softmax is already fast (BERT, DistilBert, Llama B=1).

CUTLASS handwritten GEMM+epilogue FMHA
  - long causal decoder (kind=causal, n_layer<=16): OPT S=2048, 12 layers.
    HBM fused QK+softmax+PV beats unfused bmm. XGLM is the same kind but
    24 layers — launch tax wins, stay on Triton.
  - sliding-window leftover on long S (longformer, S=1024).
  - deep stack n_layer>=30 without 4D bias (M2M100 x36): fused FA compounds
    across encoder+decoder+cross attn.
"""
from __future__ import annotations

import os

_N_LAYER = 0
_KIND = ""


def resolve(kind, n_layer):
    kind = str(kind or "")
    n_layer = int(n_layer or 0)
    if kind in ("t5_rel", "xlnet", "swin_window", "gemma_rope"):
        return "triton"
    if kind == "longformer":
        return "cutlass"
    if kind == "causal" and 1 <= n_layer <= 16:
        return "cutlass"
    # M2M100 is 36 attn; Blenderbot is 48/50 and is small-S — stay Triton.
    if 30 <= n_layer <= 40:
        return "cutlass"
    return "triton"


def set_context(n_layer=0, kind=""):
    global _N_LAYER, _KIND
    _N_LAYER = int(n_layer or 0)
    _KIND = str(kind or "")
    os.environ["GKO_ATTN_N_LAYER"] = str(_N_LAYER)
    os.environ["GKO_ATTN_KIND"] = _KIND
    forced = os.environ.get("GKO_ATTN_BACKEND", "auto").strip().lower()
    if forced in ("auto", "hybrid", ""):
        os.environ["GKO_ATTN_RESOLVED"] = resolve(_KIND, _N_LAYER)
    else:
        os.environ["GKO_ATTN_RESOLVED"] = forced


def resolved():
    r = os.environ.get("GKO_ATTN_RESOLVED", "").strip().lower()
    if r:
        return r
    return resolve(
        os.environ.get("GKO_ATTN_KIND", _KIND),
        os.environ.get("GKO_ATTN_N_LAYER", _N_LAYER) or 0,
    )
