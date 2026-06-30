from __future__ import annotations

from pathlib import Path

import pandas as pd

from ni_ai_pipeline.api_app import _add_day_after_prediction_fields
from ni_ai_pipeline.paths import PathConfig


def _paths(tmp_path: Path) -> PathConfig:
    data_bronze = tmp_path / "data" / "bronze"
    data_silver = tmp_path / "data" / "silver"
    data_gold = tmp_path / "data" / "gold"
    return PathConfig(
        root=tmp_path,
        data_bronze=data_bronze,
        data_silver=data_silver,
        data_gold=data_gold,
        bronze_dwd_dir=data_bronze / "dwd",
        bronze_lubw_dir=data_bronze / "lubw",
        bronze_messungen_dir=data_bronze / "messungen",
        silver_weather_dir=data_silver / "weather",
        silver_messungen_dir=data_silver / "messungen",
        gold_datasets_dir=data_gold / "datasets",
        site_id="default",
    )


def test_day_after_rain_prediction_is_boolean_when_prediction_is_false(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.gold_datasets_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "site_id": "default",
                "weather_time_local": "2026-06-21 12:00:00",
                "rain": 0.0,
            }
        ]
    ).to_csv(paths.gold_datasets_dir / "stuttgart_weather.csv", index=False)

    payload = _add_day_after_prediction_fields(
        {"prediction": False},
        paths=paths,
        prediction_date=pd.Timestamp("2026-06-20"),
        prediction_bool=False,
    )

    assert payload["prediction_day_after_date"] == "2026-06-21"
    assert payload["rain_prediction_day_after"] is False
    assert payload["prediction_day_after"] is False
