---
id: S3-fc__alexnet
scheme: S3-fc
case: alexnet
suite: torchbench
role: canonical
vs_naive_pct: 13.0
---

# S3-fc · alexnet

## How to use this scheme

Same S3-fc family as vgg16 (conv_frac=0.04). Here fullgraph=False + compiled Adam won; fullgraph+Adam was slightly worse. Copy the flags, not VGG's fullgraph=true.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "alexnet",
  "backend": "inductor",
  "fullgraph": false,
  "compile_optimizer": true,
  "note": "S3-fc alexnet: compiled Adam, no fullgraph",
  "actions": [
    {
      "op": "compile",
      "path": "",
      "fullgraph": false,
      "dynamic": false
    }
  ]
}
```
