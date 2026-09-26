---
id: S5-qat__resnet50_quantized_qat
scheme: S5-qat
case: resnet50_quantized_qat
suite: torchbench
role: canonical
---

# S5-qat · resnet50_quantized_qat

## How to use this scheme

QAT: compile the fp32 GraphModule. Do not wrap every FakeQuantize (old wrap −23%).
identity compile path="" is the correct usage. compiled Adam did not help here.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "resnet50_quantized_qat",
  "backend": "inductor",
  "fullgraph": false,
  "note": "S5-qat identity fp32 GraphModule",
  "actions": [
    {
      "op": "compile",
      "path": ""
    }
  ]
}
```
