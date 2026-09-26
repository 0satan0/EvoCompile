---
id: S6-fuse__addmm_gelu_template
scheme: S6-fuse
case: _addmm_gelu_template
suite: template
role: canonical
---

# S6-fuse · _addmm_gelu_template

## How to use this scheme

Only when leftover after official compile(train_step) can beat cuBLAS/cuDNN:
- extern GEMM + separate gelu: catalog addmm_gelu (CUTLASS Gemm+LinearCombinationGELU). Do NOT generate naive tl.dot (Bert 0.887×). Autograd must save pre-GELU z; recomputing addmm in bwd loses e2e train (Bert 0.80×).
- silu|mul split (SE / SwiGLU): catalog silu_mul, keep vendor conv. mobilenetv2_100 1.07×, mnasnet1_0 1.04×, tf_efficientnet_b0 1.03× vs official step.
- many triton_poi_* AND leftover triton/copy is launch/HBM bound: generated Triton poi.
NOT S6: bare cuBLAS, leftover cuDNN conv, Flash-only, fused silu_mul + leftover mm, GEMM≥60% with tiny epi.
replace_pattern then compile_step=true (no compile(model)). KernelWriter / catalog must emit def register(), custom_op gko::*, register_autograd, pass_install FX so the op is inserted at the leftover site and actually called.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "_addmm_gelu_template",
  "backend": "inductor",
  "fullgraph": false,
  "compile_step": true,
  "note": "S6: replace_pattern addmm_gelu then compile(train_step)",
  "actions": [
    {
      "op": "replace_pattern",
      "kernel": "addmm_gelu"
    }
  ]
}
```
