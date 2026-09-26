---
id: S5-opacus__opacus_cifar10
scheme: S5-opacus
case: opacus_cifar10
suite: torchbench
role: canonical
---

# S5-opacus · opacus_cifar10

## How to use this scheme

PrivacyEngine hooks: naive compile already runs. Do not skipfiles+inner fullgraph (size-assert).
identity compile path="". If you must compile, compile inner _module and leave hooks eager (recipe kind s5_opacus); this canonical file keeps identity because disable-hooks was −1%.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "opacus_cifar10",
  "backend": "inductor",
  "fullgraph": false,
  "note": "S5-opacus identity; hooks stay eager",
  "actions": [
    {
      "op": "compile",
      "path": ""
    }
  ]
}
```
