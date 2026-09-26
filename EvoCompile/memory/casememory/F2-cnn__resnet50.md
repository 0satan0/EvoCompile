---
id: F2-cnn__resnet50
scheme: F2-cnn
case: resnet50
suite: torchbench
role: canonical
---

# F2-cnn · resnet50

## How to use this scheme

True CNN, conv_frac>=0.3, fwd 0-break: naive inductor already fused the tower.
identity = compile path="" (same as vanilla). Optional tiny fullgraph is ok; do not compile_optimizer on a true CNN. If vanilla > eager at small ms, that is F2-eager skip instead.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "resnet50",
  "backend": "inductor",
  "fullgraph": false,
  "note": "F2-cnn identity; optional fullgraph",
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
