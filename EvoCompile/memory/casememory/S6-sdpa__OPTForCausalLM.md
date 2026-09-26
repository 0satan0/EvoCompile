---
id: S6-sdpa__OPTForCausalLM
scheme: S6-sdpa
case: OPTForCausalLM
suite: huggingface
role: canonical
vs_naive_pct: 24.1
---

# S6-sdpa · OPTForCausalLM

## How to use this scheme

Causal decoder, unfused bmm+softmax, no in-tree Flash. Hook OPTAttention q_proj/k_proj/v_proj to F.scaled_dot_product_attention(is_causal=True) (vendor CUTLASS FMHA) then compile_step=true. Official compile(train_step) 54.7→42.9 ms (1.28×). Do not nest compile(model). Do not start with custom Triton or handwritten tiled CUTLASS FA (~30× slower than vendor Flash). XGLM 24-layer / GPT-Neo local attn are different; T5 4D rel-bias uses fused_sdpa_attn + backend=auto (Triton tiled FA).

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "OPTForCausalLM",
  "backend": "inductor",
  "fullgraph": false,
  "compile_step": true,
  "note": "S6-sdpa: OPTAttention -> F.sdpa is_causal then compile(train_step)",
  "actions": [
    {
      "op": "apply_rewrite",
      "rewrite": "sdpa_rewrite"
    }
  ]
}
```
