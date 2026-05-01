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
- `ni-ai-pipeline train-ecoli`

## Configuration

The code reads paths from environment variables (optionally via `.env`):

- `DATA_BRONZE`, `DATA_SILVER`, `DATA_GOLD`
- `BRONZE_DWD_DIR`, `BRONZE_LUBW_DIR`, `BRONZE_MESSUNGEN_DIR`
- `SILVER_WEATHER_DIR`, `SILVER_MESSUNGEN_DIR`
- `GOLD_DATASETS_DIR`
- `SITE_ID`

If unset, defaults match the repo structure (e.g. `data/silver/weather`).
