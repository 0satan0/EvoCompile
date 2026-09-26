"""Same DistilBERT attention swap, but call in-tree F.scaled_dot_product_attention."""

from rewrites.fused_sdpa_attn import apply as _apply


def apply(model):
    return _apply(model, use_in_tree_sdpa=True)
