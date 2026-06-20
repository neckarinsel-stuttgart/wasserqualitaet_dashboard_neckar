#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if command -v ni-ai-pipeline >/dev/null 2>&1; then
  ni-ai-pipeline stuttgart-weather-hourly
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  "$REPO_ROOT/.venv/bin/python" -m ni_ai_pipeline.cli stuttgart-weather-hourly
else
  python -m ni_ai_pipeline.cli stuttgart-weather-hourly
fi
