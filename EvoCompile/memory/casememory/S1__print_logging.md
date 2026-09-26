---
id: S1__print_logging
scheme: S1
case: print_side_effect
suite: microbench
role: canonical
vs_naive_pct: 0.0
---

# S1 · print_side_effect

## How to use this scheme

When dynamo.explain shows print/logging/warnings as graph breaks and vanilla≈eager, do not AST-edit the user's forward. Emit op allow_logging (Dynamo reorderable_logging_functions), then compile path="" fullgraph=True.
That is the closed way to remove IO graph breaks. Do not invent a rewrite_class_forward kind.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "print_side_effect",
  "backend": "inductor",
  "fullgraph": false,
  "note": "S1: reorder print/logging then fullgraph",
  "actions": [
    {
      "op": "allow_logging"
    },
    {
      "op": "compile",
      "path": "",
      "fullgraph": true,
      "dynamic": false
    }
  ]
}
```
