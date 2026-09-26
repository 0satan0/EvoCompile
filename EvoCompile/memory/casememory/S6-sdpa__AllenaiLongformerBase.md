---
id: S6-sdpa__AllenaiLongformerBase
scheme: S6-sdpa
case: AllenaiLongformerBase
suite: huggingface
role: canonical
vs_naive_pct: 37.0
---

# S6-sdpa · AllenaiLongformerBase

## How to use this scheme

Sliding-window leftover on long S: official leftover is unfused bmm (softmax names often stripped). Do not copy DistilBert sdpa_rewrite. Emit kernel_spec with match_eager = local window QK/softmax/PV, hook_call = fused_sdpa_attn window arg from one_sided_attn_window_size, replace_pattern __generate__, apply_rewrite fused_sdpa_attn, compile_step=true. Landed op must be torch.ops.gko.* (not F.sdpa). Dense fused_sdpa without window is wrong.
Writer pitfalls seen in probes: (1) FX pass must use graph = gm.graph if hasattr(gm,'graph') else gm — never gm.nodes on _GMShim; (2) install_inductor_pass(attr, callable) — pass_fn is never a string; (3) custom_op returns must .clone().

## Writer snippet (USAGE ILLUSTRATION — adapt to THIS case)

```python
# USAGE ILLUSTRATION — adapt window/op name to THIS case
def _rewrite(gm):
    graph = gm.graph if hasattr(gm, 'graph') else gm
    for node in list(graph.nodes):
        ...  # match KERNEL_SPEC.fx_match only
    return gm
def register():
    install_inductor_pass('pre_grad_custom_pass', _rewrite, key=KID)
    install_post_grad_pass(_rewrite, key=KID)
# custom_op forward: return out.clone()  # never alias inputs
```

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "AllenaiLongformerBase",
  "backend": "inductor",
  "fullgraph": false,
  "compile_step": true,
  "note": "S6-sdpa: generate windowed FA + fused_sdpa_attn then compile(train_step)",
  "kernel_spec": {
    "compute": "sliding-window FA: QK^T inside one_sided window, softmax, PV",
    "modules": [
      "LongformerSelfAttention"
    ],
    "match_eager": "Eager local branch only: windowed QK^T, mask, fp32 softmax, dropout, P@V. Global-attn stays on original forward.",
    "hook_call": "fused_sdpa_attn → gko.<op>(q,k,v,bias,scale,causal,window); window=one_sided_attn_window_size; return context.clone()",
    "insert": "hook forward to torch.ops.gko.<op>; keep global-attn on original",
    "catalog_hint": null
  },
  "actions": [
    {
      "op": "replace_pattern",
      "kernel": "__generate__"
    },
    {
      "op": "apply_rewrite",
      "rewrite": "fused_sdpa_attn"
    }
  ]
}
```
