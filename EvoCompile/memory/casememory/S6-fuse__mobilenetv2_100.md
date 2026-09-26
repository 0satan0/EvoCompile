---
id: S6-fuse__mobilenetv2_100
scheme: S6-fuse
case: mobilenetv2_100
suite: timm
role: canonical
vs_naive_pct: 6.7
---

# S6-fuse · mobilenetv2_100

## How to use this scheme

SE / MBConv leftover: silu and mul in different triton_poi names. Catalog silu_mul fuses pointwise only; keep cuDNN conv. replace_pattern silu_mul then compile_step=true (no compile(model)). mobilenetv2_100 36.89→34.57 ms (1.07× vs official step); mnasnet1_0 1.04×; tf_efficientnet_b0 1.03×; mobilenet_v3_large 1.02×. Do not replace leftover cuDNN conv with CUTLASS conv.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "mobilenetv2_100",
  "backend": "inductor",
  "fullgraph": false,
  "compile_step": true,
  "note": "S6: silu_mul then compile(train_step)",
  "actions": [
    {
      "op": "replace_pattern",
      "kernel": "silu_mul"
    }
  ]
}
```
