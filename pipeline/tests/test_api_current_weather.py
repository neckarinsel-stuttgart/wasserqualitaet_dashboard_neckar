from __future__ import annotations

from pathlib import Path

import pandas as pd

from ni_ai_pipeline.api_app import _build_current_weather_payload
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


def test_current_weather_payload_returns_latest_timestamp_with_daily_max_temperature(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.gold_datasets_dir.mkdir(parents=True, exist_ok=True)

    weather_path = paths.gold_datasets_dir / "stuttgart_weather.csv"
    df = pd.DataFrame(
        [
            {
                "site_id": "default",
                "weather_time_local": "2026-06-20 12:00:00",
                "temperature_2m": 30.0,
                "wind_speed_10m": 5.0,
                "created_at_utc": "2026-06-20T10:00:00+00:00",
            },
            {
                "site_id": "default",
                "weather_time_local": "2026-06-21 09:00:00",
                "temperature_2m": 27.0,
                "wind_speed_10m": 4.0,
                "created_at_utc": "2026-06-21T07:00:00+00:00",
            },
            {
                "site_id": "default",
                "weather_time_local": "2026-06-21 12:00:00",
                "temperature_2m": 22.0,
                "wind_speed_10m": 8.0,
                "created_at_utc": "2026-06-21T10:00:00+00:00",
            },
        ]
    )

    payload = _build_current_weather_payload(df, paths=paths, weather_path=weather_path)
    assert payload["weather_time_local"] == "2026-06-21 12:00:00"
    assert payload["temperature_2m"] == 27.0
    assert payload["wind_speed_10m"] == 8.0


def test_current_weather_payload_allows_missing_wind_column(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    weather_path = paths.gold_datasets_dir / "stuttgart_weather.csv"
    df = pd.DataFrame(
        [
            {
                "site_id": "default",
                "weather_time_local": "2026-06-21 12:00:00",
                "temperature_2m": 22.0,
            }
        ]
    )

    payload = _build_current_weather_payload(df, paths=paths, weather_path=weather_path)

    assert payload["weather_time_local"] == "2026-06-21 12:00:00"
    assert payload["temperature_2m"] == 22.0
    assert "wind_speed_10m" not in payload
