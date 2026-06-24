#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Allow python -m ni_ai_pipeline.cli without editable install.
export PYTHONPATH="$REPO_ROOT/pipeline/src${PYTHONPATH:+:$PYTHONPATH}"

# Prefer containerized execution to avoid host Python dependency drift.
if command -v docker >/dev/null 2>&1; then
  if docker compose version >/dev/null 2>&1; then
    exec docker compose run --rm --no-deps pipeline ni-ai-pipeline run-daily-midnight
  elif command -v docker-compose >/dev/null 2>&1; then
    exec docker-compose run --rm --no-deps pipeline ni-ai-pipeline run-daily-midnight
  fi
fi

if command -v ni-ai-pipeline >/dev/null 2>&1; then
  ni-ai-pipeline run-daily-midnight
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  "$REPO_ROOT/.venv/bin/python" -m ni_ai_pipeline.cli run-daily-midnight
elif [[ -x "$REPO_ROOT/.venv/bin/python3" ]]; then
  "$REPO_ROOT/.venv/bin/python3" -m ni_ai_pipeline.cli run-daily-midnight
else
  /usr/bin/python3 -m ni_ai_pipeline.cli run-daily-midnight
fi
