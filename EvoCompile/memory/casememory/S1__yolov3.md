---
id: S1__yolov3
scheme: S1
case: yolov3
suite: torchbench
role: canonical
vs_naive_pct: 35.0
---

# S1 · yolov3

## How to use this scheme

Use S1 when dynamo.explain(forward) shows many hot-path breaks (setattr / nonzero / Python loop) and vanilla ≈ eager (compile never fused the tower).
yolov3: YOLOLayer.create_grids did setattr(self.nx) on the train path. rewrite_class_forward kind=yolo_layer_train_no_grid removes that, then compile path="" fullgraph=True.
Do not Titan-compile each Bottleneck (F1).

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "yolov3",
  "backend": "inductor",
  "fullgraph": false,
  "inductor": {
    "cudagraphs": false
  },
  "note": "S1: skip YOLO grid setattr, then whole Darknet fullgraph",
  "actions": [
    {
      "op": "rewrite_class_forward",
      "class": "YOLOLayer",
      "kind": "yolo_layer_train_no_grid"
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
