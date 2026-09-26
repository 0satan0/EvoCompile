---
id: S3-step__GPTNeoForCausalLM
scheme: S3-step
case: GPTNeoForCausalLM
suite: huggingface
role: canonical
vs_naive_pct: 17.6
---

# S3-step · GPTNeoForCausalLM

## How to use this scheme

0-break transformer, profiler other% (Python/sync) still ~60% after compile(model).
compile(model)+compiled Adam left that leftover; compile_step=true actions:[] wraps fwd+bwd+opt (official Dynamo boundary). Do not nest compile(model) with compile_step.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "GPTNeoForCausalLM",
  "backend": "inductor",
  "fullgraph": false,
  "compile_step": true,
  "note": "S3-step: compile(train_step) for leftover Python/opt after fused fwd",
  "actions": []
}
```
