---
id: S3-embed__MT5ForConditionalGeneration
scheme: S3-embed
case: MT5ForConditionalGeneration
suite: huggingface
role: canonical
vs_naive_pct: 16.7
---

# S3-embed · MT5ForConditionalGeneration

## How to use this scheme

Huge vocab (>=1e5) or params (>=200M) seq2seq, layerdrop_n=0 so not S2.
compiled Adam (+ optional fullgraph). Do not copy onto T5/T5Small (vocab 32k / 60M).

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "MT5ForConditionalGeneration",
  "backend": "inductor",
  "fullgraph": false,
  "compile_optimizer": true,
  "note": "S3-embed compiled Adam, n=0 not S2",
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
