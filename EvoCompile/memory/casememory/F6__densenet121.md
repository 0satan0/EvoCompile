---
id: F6__densenet121
scheme: F6
case: densenet121
suite: torchbench
role: canonical
---

# F6 · densenet121

## How to use this scheme

Many DenseLayer / _DenseLayer concat. fullgraph=True was −23%.
identity compile path="", no fullgraph.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "densenet121",
  "backend": "inductor",
  "fullgraph": false,
  "note": "F6 identity, no fullgraph",
  "actions": [
    {
      "op": "compile",
      "path": ""
    }
  ]
}
```
