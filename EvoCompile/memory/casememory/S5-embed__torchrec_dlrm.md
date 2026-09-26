---
id: S5-embed__torchrec_dlrm
scheme: S5-embed
case: torchrec_dlrm
suite: torchbench
role: canonical
---

# S5-embed · torchrec_dlrm

## How to use this scheme

CombinedOptimizer and/or EmbeddingBag: do not compile the optimizer, do not fullgraph EmbeddingBag.
identity compile path="".

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "torchrec_dlrm",
  "backend": "inductor",
  "fullgraph": false,
  "note": "S5-embed: CombinedOptimizer + EmbeddingBag identity",
  "actions": [
    {
      "op": "compile",
      "path": ""
    }
  ]
}
```
