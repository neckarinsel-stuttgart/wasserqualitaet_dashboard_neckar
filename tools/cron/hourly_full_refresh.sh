#!/usr/bin/env bash
set -euo pipefail

# Cron runs with a minimal PATH; include common locations for docker/docker-compose.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Allow python -m ni_ai_pipeline.cli without editable install.
export PYTHONPATH="$REPO_ROOT/pipeline/src${PYTHONPATH:+:$PYTHONPATH}"

# Prefer containerized execution to avoid host Python dependency drift.
if command -v docker >/dev/null 2>&1; then
  COMPOSE_CMD=""
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
  fi

  if [[ -n "$COMPOSE_CMD" ]]; then
    # Winter break (data/config/winter_break.json): skip all updates.
    # Set NI_IGNORE_WINTER_BREAK=1 to force a run. If the check itself fails, run as usual.
    if [[ "${NI_IGNORE_WINTER_BREAK:-0}" != "1" ]] \
      && $COMPOSE_CMD run --rm --no-deps -T pipeline ni-ai-pipeline winter-break --check; then
      echo "$(date -Is) winter break active, skipping refresh"
      exit 0
    fi

    $COMPOSE_CMD run --rm --no-deps crawl_dwd
    $COMPOSE_CMD run --rm --no-deps crawl_lubw
    exec $COMPOSE_CMD run --rm --no-deps pipeline ni-ai-pipeline run-api-refresh
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

if [[ "${NI_IGNORE_WINTER_BREAK:-0}" != "1" ]] \
  && "$PYTHON" -m ni_ai_pipeline.cli winter-break --check; then
  echo "$(date -Is) winter break active, skipping refresh"
  exit 0
fi

"$PYTHON" "$REPO_ROOT/scripts/bronze/crawl_dwd.py"
"$PYTHON" "$REPO_ROOT/scripts/bronze/crawl_lubw.py"

if command -v ni-ai-pipeline >/dev/null 2>&1; then
  ni-ai-pipeline run-api-refresh
else
  "$PYTHON" -m ni_ai_pipeline.cli run-api-refresh
fi
