---
id: S3-deep__XLNetLMHeadModel
scheme: S3-deep
case: XLNetLMHeadModel
suite: huggingface
role: canonical
vs_naive_pct: 53.3
---

# S3-deep · XLNetLMHeadModel

## How to use this scheme

Deep leftover (n_layer>=20 or HF_LLM) with vanilla_ms typically >=80 and Adam.
compile_optimizer=true; extra fullgraph only if LayerDrop n>=1 (then this overlaps S2).
HF_LLM: eval with --no-amp. Do not use on MegatronBert whose vanilla_ms is already <80 (F2-skinny).

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "XLNetLMHeadModel",
  "backend": "inductor",
  "fullgraph": false,
  "compile_optimizer": true,
  "note": "S3-deep compiled Adam; LayerDrop off if n>=1",
  "actions": [
    {
      "op": "hf_layerdrop_off"
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
