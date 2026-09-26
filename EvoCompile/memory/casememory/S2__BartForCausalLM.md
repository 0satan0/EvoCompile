---
id: S2__BartForCausalLM
scheme: S2
case: BartForCausalLM
suite: huggingface
role: canonical
vs_naive_pct: 10.7
---

# S2 · BartForCausalLM

## How to use this scheme

Use S2 only when inspect prints layerdrop_n >= 1. n=0 (T5, Distil) is not S2.
hf_layerdrop_off first (must run before compile), then compile path="" fullgraph=True, and compile_optimizer=true for leftover Adam launches.
Do not copy this JSON onto DistilBert.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "BartForCausalLM",
  "backend": "inductor",
  "fullgraph": false,
  "inductor": {
    "cudagraphs": false
  },
  "compile_optimizer": true,
  "note": "S2: hf_layerdrop_off then fullgraph + compiled Adam",
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
