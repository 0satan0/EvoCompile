---
id: identity__microbench
scheme: identity
case: microbench_unbacked_tolist_sum
suite: torchbench
role: canonical
---

# identity · microbench_unbacked_tolist_sum

## How to use this scheme

Fallback when no accel leaf hits (or vanilla_ms<5 / n_params<0.2).
identity is compile path="" — the same as naive torch.compile(model). Do not skip compile unless F2-eager (vanilla > eager).

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "microbench_unbacked_tolist_sum",
  "backend": "inductor",
  "fullgraph": false,
  "note": "identity fallback = naive compile(model)",
  "actions": [
    {
      "op": "compile",
      "path": ""
    }
  ]
}
```
