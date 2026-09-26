# GKO Compile Agent (slim)

Minimal source for the multi-agent `torch.compile` / Inductor optimization loop.

**97 files / ~1.1MB** — only modules on the `python -m agent.loop` call path, plus expert memory and catalog kernels/rewrites. No historical recipes, no generated per-case kernels, no tests, no ablation harness, no `oldversion` / `sever`.

Largest non-code chunk: `memory/casememory/` (~32 scheme usage cards attached at retrieve). Core Python is ~50 files.

## What runs

```
Collector → Retriever → Similarity → Optimizer (+ RewriteWriter / TritonWriter)
         → Evaluator → Repair / Evolve
```

```bash
export GKO="$PWD"
source docker/gko_env.sh
python -m agent.loop --case DistilBertForMaskedLM --suite huggingface \
  --llm-url http://127.0.0.1:8000/v1 --llm-model your-model \
  --max-rounds 3 --out-dir run/gko_eval/demo
```

API key via env only: `GKO_LLM_API_KEY` or `--llm-api-key`. Never commit keys.

## Layout

| Path | Role |
|------|------|
| `agent/` | Loop + 9 agents + LLM client + prompts / fingerprint / residual |
| `agent/agents/` | collector, retriever, similarity, generator, writers, evaluator, repairer, evolver |
| `eval/recipe.py` | Apply recipe JSON (Layer A/B/C) |
| `eval/run_agent_cases.py` | Load workload + time eager/vanilla/agent |
| `eval/inspect_features.py` / `trace_dump.py` | Collector sensors |
| `eval/kernels/` | Catalog only (`addmm_gelu`, `fused_sdpa`, `silu_mul`, …); `generated/` empty at ship |
| `eval/rewrites/` | Catalog Layer-B hooks; `generated/` empty at ship |
| `memory/treememory` | Hard retrieve tree |
| `memory/casememory/` | Scheme usage cards attached at retrieve |
| `memory/scheme_index` | Human-readable scheme catalog |
| `docs/` | I/O + evidence notes |

## Three layers (L1/L2/L3 ≈ A/B/C)

- **A / L1**: compile region & knobs (`compile path=`, `fullgraph`, `compile_step`, …)
- **B / L2**: method patch (`apply_rewrite`, RewriteWriter → `eval/rewrites/generated/`)
- **C / L3**: custom kernel (`replace_pattern`, TritonWriter → `eval/kernels/generated/`)

## Intentionally excluded

- `eval/recipes/**` (past landed JSON library)
- `eval/kernels/generated/*`, `eval/rewrites/generated/*`, `rewrites/s6_28`, probes
- `agent/ablation.py`, `agent/test_*.py`, `agent/oldversion/`, `agent/sever/`
- Cross-backend runners, suite CSVs, logs, API keys

Generated B/C files appear under `eval/*/generated/` only after the agent writes them at runtime.

## Rebuild retrieve tree

```bash
python -m agent.memory_bootstrap
```
