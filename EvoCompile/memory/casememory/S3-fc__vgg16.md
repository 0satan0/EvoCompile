---
id: S3-fc__vgg16
scheme: S3-fc
case: vgg16
suite: torchbench
role: canonical
vs_naive_pct: 5.0
---

# S3-fc · vgg16

## How to use this scheme

Named CNN but conv_param_frac<0.3, Linear dominates, Adam, profiler other% (sync) high.
This is leftover classifier + Adam, not a fused conv tower. compile_optimizer=true.
vgg prefix may keep fullgraph=true; alexnet uses fullgraph=false. Do not treat as F2-cnn identity.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "vgg16",
  "backend": "inductor",
  "fullgraph": false,
  "compile_optimizer": true,
  "note": "S3-fc compiled Adam; keep fullgraph on VGG",
  "actions": [
    {
      "op": "compile",
      "path": "",
      "fullgraph": true,
      "dynamic": false
    }
  ]
}
```
