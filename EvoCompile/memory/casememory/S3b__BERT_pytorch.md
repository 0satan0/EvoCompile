---
id: S3b__BERT_pytorch
scheme: S3b
case: BERT_pytorch
suite: torchbench
role: canonical
vs_naive_pct: 5.0
---

# S3b · BERT_pytorch

## How to use this scheme

0-break + Adam but vanilla already 1.20–2.20× eager (or setattr/.item). fullgraph=True was −6.9% on this case.
Use compile path="" fullgraph=false + compile_optimizer=true. Do not copy BertForMaskedLM fullgraph.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "BERT_pytorch",
  "backend": "inductor",
  "fullgraph": false,
  "compile_optimizer": true,
  "note": "S3b: no fullgraph + compiled Adam",
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
