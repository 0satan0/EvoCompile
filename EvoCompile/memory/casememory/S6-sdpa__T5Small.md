---
id: S6-sdpa__T5Small
scheme: S6-sdpa
case: T5Small
suite: huggingface
role: canonical
---

# S6-sdpa · T5Small

## How to use this scheme

T5Attention leftover: unfused bmm + 4D relative-position bias + fp32 softmax. Do not copy DistilBert sdpa_rewrite (F.sdpa drops the bias). Emit kernel_spec match_eager for scores+bias→softmax→dropout→P@V with scale=1.0 (scale in q weights), modules=['T5Attention'], replace_pattern __generate__, fused_sdpa_attn, compile_step. custom_op MUST return .clone() — aliasing inputs crashes torch 2.4. Repair of alias/schema is Layer C, even if the stack passes through fused_sdpa_attn.py.

## Writer snippet (USAGE ILLUSTRATION — adapt to THIS case)

```python
# USAGE ILLUSTRATION — T5 rel-bias contract
@torch.library.custom_op('gko::t5_rel_attn', mutates_args=())
def t5_rel_attn(q, k, v, bias, scale: float, causal: int, window: int):
    # match_eager: Q@K^T (+bias), fp32 softmax, dropout, P@V; scale usually 1.0
    out = ...
    return out.clone()  # required — no input alias
```

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "T5Small",
  "backend": "inductor",
  "fullgraph": false,
  "compile_step": true,
  "note": "S6-sdpa: generate T5 rel-bias FA + fused_sdpa_attn then compile(train_step)",
  "kernel_spec": {
    "compute": "T5 attn with 4D relative bias before softmax",
    "modules": [
      "T5Attention"
    ],
    "match_eager": "After Q/K/V proj: scores=Q@K^T (scale=1.0), add 4D position_bias, fp32 softmax+cast, dropout, P@V",
    "hook_call": "fused_sdpa_attn._call → gko.<op>; return context.clone()",
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
