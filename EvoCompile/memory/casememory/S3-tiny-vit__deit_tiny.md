---
id: S3-tiny-vit__deit_tiny
scheme: S3-tiny-vit
case: deit_tiny_patch16_224.fb_in1k
suite: timm
role: canonical
vs_naive_pct: 13.6
---

# S3-tiny-vit · deit_tiny_patch16_224.fb_in1k

## How to use this scheme

ViT embed_dim<=192 (DeiT-tiny), Adam, leftover 1.2–1.8x.
fullgraph + compile_optimizer. Do not copy onto DeiT-base / Swin / CLIP (those are already fused → identity).

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "deit_tiny_patch16_224.fb_in1k",
  "backend": "inductor",
  "fullgraph": false,
  "compile_optimizer": true,
  "note": "S3-tiny-vit fullgraph + compiled Adam",
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
