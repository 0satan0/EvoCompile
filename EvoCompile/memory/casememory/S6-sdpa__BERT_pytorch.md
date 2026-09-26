---
id: S6-sdpa__BERT_pytorch
scheme: S6-sdpa
case: BERT_pytorch
suite: torchbench
role: canonical
---

# S6-sdpa · BERT_pytorch

## How to use this scheme

BERT_pytorch Attention already has projected QKV [B,H,S,D]. Fuse score/softmax/PV only. Hook builds additive bias via _bias_keep_mask — your op receives bias broadcastable to scores, NOT the raw Boolean mask. kernel_spec.hook_call must say that. replace_pattern __generate__ + fused_sdpa_attn + compile_step. Do not switch to compile(model) on Repair. If crash is inside fused_sdpa_attn._bias_keep_mask under Dynamo, that is Layer B (hook helper); if gko:: op shape/rel_loss, that is Layer C.

## Writer snippet (USAGE ILLUSTRATION — adapt to THIS case)

```python
# USAGE ILLUSTRATION — already-projected QKV + hook bias
# hook: bias = _bias_keep_mask(mask, bs, k_len, ...); then op(q,k,v,bias,scale,...)
# WRONG: re-apply mask.eq(0) inside the op with a different broadcast
def forward(q, k, v, bias, scale, causal, window):
    scores = (q @ k.transpose(-2, -1)) * scale
    if bias is not None:
        scores = scores + bias  # already additive keep-mask layout
    ...
    return ctx.clone()
```

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "BERT_pytorch",
  "backend": "inductor",
  "fullgraph": false,
  "compile_step": true,
  "note": "S6-sdpa: generate BERT_pytorch FA + fused_sdpa_attn then compile(train_step)",
  "kernel_spec": {
    "compute": "masked self-attn on projected QKV",
    "modules": [
      "Attention",
      "MultiHeadedAttention"
    ],
    "match_eager": "scores=(Q@K^T)*scale, add keep-mask bias, softmax, dropout, P@V; return context",
    "hook_call": "_bert_pytorch_attn builds bias via _bias_keep_mask then _call; op must accept that bias layout",
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
