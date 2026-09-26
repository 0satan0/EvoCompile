---
id: S5+S3__demucs
scheme: S5+S3
case: demucs
suite: torchbench
role: canonical
---

# S5+S3 · demucs

## How to use this scheme

LSTM/cuDNN is an intentional graph break (S5: do not compile LSTM into Inductor). If the step is still long and Adam leftover is high, still compile(model)+compile_optimizer.
actions compile path="" fullgraph=false, compile_optimizer=true.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "demucs",
  "backend": "inductor",
  "fullgraph": false,
  "compile_optimizer": true,
  "inductor": {
    "cudagraphs": false
  },
  "note": "S5 LSTM eager + compiled Adam",
  "actions": [
    {
      "op": "compile",
      "path": ""
    }
  ]
}
```
