---
id: S3-step__GoogleFnet
scheme: S3-step
case: GoogleFnet
suite: huggingface
role: canonical
vs_naive_pct: 5.9
---

# S3-step · GoogleFnet

## How to use this scheme

compile_step=true with actions: [] is torch.compile(train_step): one graph over fwd+bwd+opt. Naive baseline stays compile(model). Use when leftover after compile(model) is train_step Python / opt.step / a vendor fallback (cuFFT), not a hot-path rewrite (S1) and not moco/contrastive.
FNet FFT is the canonical vendor-fallback instance: do not fullgraph the FFT and do not only compile Adam.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "GoogleFnet",
  "backend": "inductor",
  "fullgraph": false,
  "compile_step": true,
  "note": "S3-step: compile(train_step), FFT stays cuFFT",
  "actions": []
}
```
