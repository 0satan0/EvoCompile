#!/usr/bin/env bash
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GKO="${GKO:-${_HERE}}"
source "${GKO}/docker/gko_env.sh"
PY="${GKO_AGENT_PY:-${PY:-python3}}"
OUT="${GKO_AGENT_OUT:-${GKO}/run/gko_eval/agent_optico}"
mkdir -p "$OUT"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$GKO"
CASES=(
  optico-llama2-0.5b
  optico-llama3-0.5b
  optico-qwen25-0.5b
  optico-llama2-7b
  optico-qwen25-7b
  optico-llama3-8b
  optico-qwen3-moe-a0.6b
)
for c in "${CASES[@]}"; do
  echo "[agent] $c" | tee -a "$OUT/agent.suite.log"
  set +e
  "$PY" -m agent.loop --case "$c" --suite huggingface --no-llm \
    --warmup 6 --measure 20 --max-rounds 3 \
    --out-dir "$OUT/$c" \
    2>&1 | tee -a "$OUT/agent.suite.log"
  echo "[agent] $c exit=$?" | tee -a "$OUT/agent.suite.log"
  set -e
done
echo AGENT_OPTICO_FINISHED | tee -a "$OUT/agent.suite.log"
