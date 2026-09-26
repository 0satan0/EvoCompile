---
id: F2-skinny__DistilBertForMaskedLM
scheme: F2-skinny
case: DistilBertForMaskedLM
suite: huggingface
role: canonical
---

# F2-skinny · DistilBertForMaskedLM

## How to use this scheme

model_type contains distil, or n_layer<=6 with vanilla already >=2.3x, or hidden in (0,256], or n_tensors>500, or n_layer>=20 but vanilla_ms<80.
identity compile path="". Old S3/S2 JSON was DistilBert 66.7→112 (−68%). Missing hidden_size is NOT skinny (that used to steal BERT_pytorch).

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "DistilBertForMaskedLM",
  "backend": "inductor",
  "fullgraph": false,
  "note": "F2-skinny identity = naive compile(model)",
  "actions": [
    {
      "op": "compile",
      "path": ""
    }
  ]
}
```
