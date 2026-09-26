---
id: S4__detectron2_maskrcnn_r_50_c4
scheme: S4
case: detectron2_maskrcnn_r_50_c4
suite: torchbench
role: canonical
---

# S4 · detectron2_maskrcnn_r_50_c4

## How to use this scheme

Only when vanilla compile FAILED. Leave Python/dynamic/Polygon/Instances eager; compile the static backbone.
disable_forward on the parent, compile path=backbone, skip_code_on_types for PolygonMasks/Instances/Boxes.
If naive compile already runs (vision_maskrcnn), do not use S4 — that is identity.

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "detectron2_maskrcnn_r_50_c4",
  "backend": "inductor",
  "fullgraph": false,
  "inductor": {
    "cudagraphs": false
  },
  "note": "S4: backbone only; RPN/ROI eager",
  "actions": [
    {
      "op": "skip_code_on_types",
      "targets": [
        "detectron2.structures.masks.PolygonMasks.__getitem__",
        "detectron2.structures.instances.Instances.__getitem__",
        "detectron2.structures.boxes.Boxes.__getitem__"
      ]
    },
    {
      "op": "disable_forward",
      "path": ""
    },
    {
      "op": "compile",
      "path": "backbone",
      "dynamic": false
    },
    {
      "op": "disable_forward",
      "path": "proposal_generator"
    },
    {
      "op": "disable_forward",
      "path": "roi_heads"
    },
    {
      "op": "disable_method",
      "path": "roi_heads",
      "name": "label_and_sample_proposals"
    }
  ]
}
```
