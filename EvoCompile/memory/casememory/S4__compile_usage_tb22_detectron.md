---
id: S4__compile_usage_tb22_detectron
scheme: S4
case: detectron2_*_tb22
suite: torchbench
role: usage_fix
---

# S4 · TB22 Detectron / generate compile-usage fix (no LLM)

## Symptom (eager OK, vanilla / official Dynamo FAIL)

- Official inductor warmup: `NameError: ToFloat` / `FloatTrueDiv` (FxGraphCache / symbolic).
- Naive `torch.compile(model)`: Dynamo fails on `PolygonMasks` / `Instances` / dynamic ROI paths
  (error text often truncated around `ygons[i] for i in item]`).

## Root cause

Not primarily an env/weight issue once `.pkl` is present. Whole-model compile pulls Python/dynamic
detectron2 structures into the graph. Official CUDA training path is fine for eager.

## Fix (fixed recipe — apply before measuring)

1. Clear inductor/dynamo caches if ToFloat persists: `rm -rf $TMPDIR/torchinductor_*` and
   `torch._dynamo.reset()`.
2. `inductor.cudagraphs = false`.
3. `skip_code_on_types` for PolygonMasks / Instances / Boxes `__getitem__`.
4. `disable_forward` on parent, `compile` only `path=backbone`, disable `proposal_generator` /
   `roi_heads` (and `label_and_sample_proposals`).

Canonical JSON: `eval/recipes/detectron2_*.json` (same actions as
`eval/recipes/detectron2_maskrcnn_r_50_c4.json`).

## Measured (S4_FIX3, host 68, AMP, warmup=4 measure=10)

| model | eager_ms | S4/recipe_ms | recipe/eager | vanilla_ok |
|-------|----------|--------------|--------------|------------|
| detectron2_fasterrcnn_r_50_c4 | 105.23 | 177.53 | 0.59× | false |
| detectron2_fasterrcnn_r_50_dc5 | 153.14 | 142.85 | 1.07× | false |
| detectron2_fasterrcnn_r_50_fpn | 174.29 | 157.95 | 1.10× | false |
| detectron2_fasterrcnn_r_101_c4 | 204.59 | 195.78 | 1.05× | false |
| detectron2_fasterrcnn_r_101_dc5 | 184.12 | 158.38 | 1.16× | false |
| detectron2_fasterrcnn_r_101_fpn | 164.64 | 99.84 | 1.65× | false |
| detectron2_maskrcnn_r_50_fpn | 219.96 | 141.40 | 1.56× | false |
| detectron2_maskrcnn_r_101_c4 | 240.07 | 210.22 | 1.14× | false |
| detectron2_maskrcnn_r_101_fpn | 233.46 | 238.17 | 0.98× | false |
| llama (nocudagraph) | 11.23 | 6.52 | 1.72× | true (van 1.85×) |
| cm3leon_generate (decoder-only) | 711.32 | 575.16 | 1.24× | skipped |

Out: `run/gko_eval/tb22_s4_fix_20260923_002846/`.

## Related compile-usage (not S4 backbone)

- `llama`: CUDAGraphs overwrite → `llama_nocudagraph.json` (compile model, cudagraphs off).
- `cm3leon_generate`: CUDAGraphs overwrite **plus** autoregressive seq growth.
  Whole-model `dynamic=false` → endless `cache_size_limit` recompiles; whole-model
  `dynamic=true` → multi-10min inductor hang. Recipe: compile only `path=model`
  (TransformerDecoder), leave `SequenceGenerator` loop eager; `dynamic=false`,
  `cache_size_limit=128`, cudagraphs off. Prefer `--skip-vanilla`.
- `doctr_*` / `sam_fast` / `pyhpc_*` / `moondream` / `llama_v2_7b_16h` / `maml` /
  `simple_gpt` / `detectron2_fcos_r_50_fpn`: remaining-11 recipes under
  `eval/recipes/<name>.json` (fcos/sam_fast = S4-style; others identity + cudagraphs off).
  Runner: `run/gko_eval/_run_tb22_remain11_68.sh`.
- Official TB22 3-mode: `run/dynamo_tb22_rerun_*` (env deps must be present; Detectron
  ToFloat/CUDAGraphs fails are compile-usage, not missing-package).
