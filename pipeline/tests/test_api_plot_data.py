from __future__ import annotations

from pathlib import Path

import pandas as pd

from ni_ai_pipeline.api_app import _load_gold_plot_dataframe
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


def test_plot_dataframe_fills_sparse_wq_for_newer_weather_dates(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.gold_datasets_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"site_id": "default", "date": "2026-06-28", "SD_SO_sum": None},
            {"site_id": "default", "date": "2026-06-29", "SD_SO_sum": None},
        ]
    ).to_csv(paths.gold_datasets_dir / "gold_daily_features.csv", index=False)

    pd.DataFrame(
        [
            {
                "zeit": "2026-06-27",
                "We_Ne_ElektrischeLeitfaehigkeit_mean": 899.9,
                "We_Ne_Sauerstoff_mean": 12.5,
                "Ho_Ne_Truebung,quantitativ_mean": 8.4,
                "We_Ne_pH-Wert_mean": 8.3,
            }
        ]
    ).to_csv(paths.gold_datasets_dir / "masterdata.csv", index=False)

    out = _load_gold_plot_dataframe(paths)

    assert out["SD_SO_hours_day"].tolist() == [0.0, 0.0]
    assert out["We_Ne_ElektrischeLeitfaehigkeit"].tolist() == [899.9, 899.9]
    assert out["Sauerstoff"].tolist() == [12.5, 12.5]
    assert out["Ho_Ne_Truebung,quantitativ"].tolist() == [8.4, 8.4]
    assert out["pH_wert"].tolist() == [8.3, 8.3]
