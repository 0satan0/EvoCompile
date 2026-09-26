---
id: S3-unfused__BertForMaskedLM
scheme: S3-unfused
case: BertForMaskedLM
suite: huggingface
role: canonical
vs_naive_pct: 5.6
---

# S3-unfused · BertForMaskedLM

## How to use this scheme

0-break transformer, standard Adam, vanilla/eager <= 1.20, hidden>=768, >=12 real transformer layers.
BertLayer is already one graph; remaining gain is compiled Adam plus fullgraph to fuse leftover pointwise.
actions: compile path="" fullgraph=true AND compile_optimizer=true.
Do not use this on DistilBert (6 layers, vanilla already 2.6x) or BERT_pytorch (vanilla already 1.2–2.2x → S3b).

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "BertForMaskedLM",
  "backend": "inductor",
  "fullgraph": false,
  "compile_optimizer": true,
  "dynamo": {
    "capture_scalar_outputs": true
  },
  "note": "S3-unfused: fullgraph + compiled Adam",
  "actions": [
    {
      "op": "compile",
      "path": "",
      "fullgraph": true,
      "dynamic": false
    }
  ]
}
```
