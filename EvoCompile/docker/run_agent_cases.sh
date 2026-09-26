#!/usr/bin/env bash
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GKO="${GKO:-${_HERE}}"
source "${GKO}/docker/gko_env.sh"
cd "${GKO}/eval"
exec "${PY:-python3}" run_agent_cases.py "$@"
