---
id: S6-sdpa__DistilBertForMaskedLM
scheme: S6-sdpa
case: DistilBertForMaskedLM
suite: huggingface
role: canonical
vs_naive_pct: 16.3
---

# S6-sdpa · DistilBertForMaskedLM

## How to use this scheme

I-channel: extern bmm + triton softmax, no _scaled_dot_product_flash/efficient. Class is DistilBERT MultiHeadSelfAttention (q_lin/k_lin/v_lin), AMP fp16, S=128, head_dim=64.
First leaf stays F2-skinny identity (S2/S3 Adam was DistilBert −68%). After identity, apply_rewrite sdpa_rewrite then compile_step=true (official train_step wrap). Do not also compile(model). In-tree F.sdpa + compile(train_step) 53.10→49.36 ms (+7.6% vs official step; ~16% vs naive compile(model)). Custom Triton fused_sdpa is not the first retrieve — vendor SDPA has the fused bwd.
Also lands on OPTForCausalLM (causal is_causal / vendor FMHA). Longformer / T5 / 4D rel-bias use fused_sdpa_attn + fused_sdpa backend=auto, not this DistilBert rewrite.
Do not overwrite eval/recipes/DistilBertForMaskedLM.json (gold identity).

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "DistilBertForMaskedLM",
  "backend": "inductor",
  "fullgraph": false,
  "compile_step": true,
  "note": "S6-sdpa: MultiHeadSelfAttention -> F.sdpa then compile(train_step)",
  "actions": [
    {
      "op": "apply_rewrite",
      "rewrite": "sdpa_rewrite"
    }
  ]
}
```
