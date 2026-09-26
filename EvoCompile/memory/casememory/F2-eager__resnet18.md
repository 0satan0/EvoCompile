---
id: F2-eager__resnet18
scheme: F2-eager
case: resnet18
suite: torchbench
role: canonical
vs_naive_pct: 18.0
---

# F2-eager · resnet18

## How to use this scheme

vanilla torch.compile(model) is slower than eager on a small CNN step (~10ms).
The correct recipe is skip compile: actions: [] (NOT compile path="", which is identity=vanilla).
Do not compile_optimizer. Do not copy this skip onto mv3/densenet where vanilla is already faster.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "resnet18",
  "backend": "inductor",
  "fullgraph": false,
  "note": "F2-eager: skip compile, vanilla > eager",
  "actions": []
}
```
