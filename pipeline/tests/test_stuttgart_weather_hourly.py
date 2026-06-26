from __future__ import annotations

from pathlib import Path

import pandas as pd

from ni_ai_pipeline.paths import PathConfig
from ni_ai_pipeline.steps.gold_stuttgart_weather import pull_stuttgart_weather_hourly


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict:
        return self._payload


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
        site_id="stuttgart",
    )


def test_pull_stuttgart_weather_hourly_selects_requested_hour(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    payload = {
        "hourly": {
            "time": ["2026-06-20T13:00", "2026-06-20T14:00", "2026-06-20T15:00"],
            "temperature_2m": [20.0, 21.0, 22.0],
            "relative_humidity_2m": [60, 55, 50],
            "dew_point_2m": [12.0, 12.1, 12.2],
            "apparent_temperature": [20.0, 21.1, 22.2],
            "wind_speed_10m": [5.2, 6.1, 5.8],
            "precipitation_probability": [10, 20, 30],
            "precipitation": [0.0, 0.1, 0.0],
            "rain": [0.0, 0.1, 0.0],
            "showers": [0.0, 0.0, 0.0],
            "weather_code": [1, 2, 3],
        }
    }

    def _fake_get(*args, **kwargs):
        return _FakeResponse(payload)

    monkeypatch.setattr("ni_ai_pipeline.steps.gold_stuttgart_weather.requests.get", _fake_get)

    out_path = pull_stuttgart_weather_hourly(paths, requested_hour=14, timezone="Europe/Berlin")

    out = pd.read_csv(out_path)
    assert len(out) == 1
    assert str(out.loc[0, "weather_time_local"]).startswith("2026-06-20 14:00:00")
    assert float(out.loc[0, "temperature_2m"]) == 21.0
    assert float(out.loc[0, "wind_speed_10m"]) == 6.1

    pull_stuttgart_weather_hourly(paths, requested_hour=14, timezone="Europe/Berlin")
    out2 = pd.read_csv(out_path)
    assert len(out2) == 1
