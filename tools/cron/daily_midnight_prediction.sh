#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Allow python -m ni_ai_pipeline.cli without editable install.
export PYTHONPATH="$REPO_ROOT/pipeline/src${PYTHONPATH:+:$PYTHONPATH}"

# Prefer containerized execution to avoid host Python dependency drift.
if command -v docker >/dev/null 2>&1; then
  if docker compose version >/dev/null 2>&1; then
    docker compose run --rm --no-deps crawl_dwd
    docker compose run --rm --no-deps crawl_lubw
    exec docker compose run --rm --no-deps pipeline ni-ai-pipeline run-daily-midnight
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose run --rm --no-deps crawl_dwd
    docker-compose run --rm --no-deps crawl_lubw
    exec docker-compose run --rm --no-deps pipeline ni-ai-pipeline run-daily-midnight
  fi
fi

# Non-Docker fallback: resolve Python once, then run crawlers before the pipeline.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python3" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python3"
else
  PYTHON="/usr/bin/python3"
fi

"$PYTHON" "$REPO_ROOT/scripts/bronze/crawl_dwd.py"
"$PYTHON" "$REPO_ROOT/scripts/bronze/crawl_lubw.py"

if command -v ni-ai-pipeline >/dev/null 2>&1; then
  ni-ai-pipeline run-daily-midnight
else
  "$PYTHON" -m ni_ai_pipeline.cli run-daily-midnight
fi
