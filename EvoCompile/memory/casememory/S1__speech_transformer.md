---
id: S1__speech_transformer
scheme: S1
case: speech_transformer
suite: torchbench
role: canonical
vs_naive_pct: 5.2
---

# S1 · speech_transformer

## How to use this scheme

Same family as yolov3: breaks are inside pad-mask / Decoder.preprocess, not between fused layers.
speech_vectorize_pad_mask + decoder_preprocess_tensor + no-inplace encoder/decoder + sdp_attn_no_numpy_inf, then one fullgraph compile of the whole Transformer.
capture_scalar_outputs helps leftover .item().

## Recipe (apply with eval/recipe.py apply_recipe)

```json
{
  "model": "speech_transformer",
  "backend": "inductor",
  "fullgraph": false,
  "inductor": {
    "cudagraphs": false
  },
  "dynamo": {
    "capture_scalar_outputs": true
  },
  "note": "S1 speech: vectorize pad-mask then fullgraph",
  "actions": [
    {
      "op": "speech_vectorize_pad_mask"
    },
    {
      "op": "rewrite_class_forward",
      "class": "Decoder",
      "kind": "decoder_preprocess_tensor"
    },
    {
      "op": "rewrite_class_forward",
      "class": "EncoderLayer",
      "kind": "encoder_layer_no_inplace"
    },
    {
      "op": "rewrite_class_forward",
      "class": "DecoderLayer",
      "kind": "decoder_layer_no_inplace"
    },
    {
      "op": "rewrite_class_forward",
      "class": "ScaledDotProductAttention",
      "kind": "sdp_attn_no_numpy_inf"
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
