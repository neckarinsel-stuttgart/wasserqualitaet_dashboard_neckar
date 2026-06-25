# ni-ai-pipeline

This is the **production Python subproject** for the NI_AI repository.

Goal: keep Jupyter notebooks for exploration, but run/ship the data pipeline and model training as importable, testable `.py` code.

## What this mirrors

Notebook orchestration in the repo currently runs:
- Bronze: `scripts/bronze/crawl_dwd.ipynb`, `scripts/bronze/crawl_lubw.ipynb`
- Silver: `scripts/silver/clean_dwd_daten.ipynb`, `scripts/silver/create_messungen_complete.ipynb`
- Gold: `scripts/gold/data_full.ipynb`, `scripts/gold/create_masterdata.ipynb`, `scripts/gold/build_daily_gold_dataset.ipynb`

This package ports the **Silver + Gold** steps and the **ecoli training/evaluation** into Python modules.

## Install (dev)

From repo root:

- `python -m pip install -e ./pipeline[dev,train]`

## Run

- `ni-ai-pipeline --help`
- `ni-ai-pipeline run-all`

You can also run step-by-step:
- `ni-ai-pipeline silver-weather`
- `ni-ai-pipeline silver-messungen`
- `ni-ai-pipeline gold-data-full`
- `ni-ai-pipeline gold-masterdata`
- `ni-ai-pipeline gold-daily-dataset`
- `ni-ai-pipeline stuttgart-weather-hourly`
- `ni-ai-pipeline train-ecoli`
- `ni-ai-pipeline predict-daily`
- `ni-ai-pipeline run-daily-midnight`

`train-ecoli` now persists a reusable model artifact by default:
- `ecoli_model.pkl`
- `ecoli_model_metadata.json`

You can control this behavior:
- `ni-ai-pipeline train-ecoli --no-save-model`
- `ni-ai-pipeline train-ecoli --model-path <path> --model-metadata-path <path>`

Daily prediction flow (for cron):
- `predict-daily` loads latest row from `gold_daily_features.csv`, runs the saved model, and writes/upserts `predictions.csv`.
- `run-daily-midnight` refreshes Silver+Gold outputs and then runs `predict-daily`.

Hourly Stuttgart weather flow (for cron):
- `stuttgart-weather-hourly` fetches all hourly weather values for today from Open-Meteo, selects one requested hour (default: current local hour), and writes/upserts one row into `stuttgart_weather.csv`.
- Example: `ni-ai-pipeline stuttgart-weather-hourly --hour 14`
- Example cron entry (every hour): `0 * * * * /path/to/repo/tools/cron/stuttgart_weather_hourly.sh >> /path/to/repo/logs/stuttgart_weather_hourly.log 2>&1`

API endpoint prerequisites:
- `/get_current_weather` reads `GOLD_DATASETS_DIR/stuttgart_weather.csv` (or default `data/gold/datasets/stuttgart_weather.csv`).
- `/get_daily_prediction` reads `GOLD_DATASETS_DIR/predictions.csv` (or default `data/gold/datasets/predictions.csv`).
- `/get_last_30d_weather` returns all weather rows from the last 30 calendar days.
- `/get_last_30d_predictions` returns all prediction rows from the last 30 calendar days.
- Alias routes also available: `/get-last-30d-weather` and `/get-last-30d-predictions`.
- If these files are missing in deployment, run:
	- `ni-ai-pipeline stuttgart-weather-hourly`
	- `ni-ai-pipeline predict-daily` (requires `ecoli_model.pkl` and `ecoli_model_metadata.json` in the same datasets directory)

First-deploy bootstrap behavior:
- `predict-daily` now auto-trains once when default model artifacts are missing, then writes `predictions.csv`.
- `run-daily-midnight` also auto-trains once on missing model artifacts before prediction.

Example cron entry (Linux):
- `0 0 * * * /path/to/repo/tools/cron/daily_midnight_prediction.sh >> /path/to/repo/logs/daily_midnight_prediction.log 2>&1`

## Configuration

The code reads paths from environment variables (optionally via `.env`):

- `DATA_BRONZE`, `DATA_SILVER`, `DATA_GOLD`
- `BRONZE_DWD_DIR`, `BRONZE_LUBW_DIR`, `BRONZE_MESSUNGEN_DIR`
- `SILVER_WEATHER_DIR`, `SILVER_MESSUNGEN_DIR`
- `GOLD_DATASETS_DIR`
- `SITE_ID`

If unset, defaults match the repo structure (e.g. `data/silver/weather`).
