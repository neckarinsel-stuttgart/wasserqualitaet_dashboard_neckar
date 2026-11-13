from __future__ import annotations

from pathlib import Path

from ni_ai_pipeline.model_registry import append_model_version, read_latest_model_version
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


def test_registry_starts_at_one_and_counts_up(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    assert append_model_version(
        paths, trained_at="2026-10-08T16:17:12+00:00", training_data_until="2026-10-05"
    ) == 1
    assert append_model_version(
        paths, trained_at="2026-11-20T08:00:00+00:00", training_data_until="2026-11-18"
    ) == 2

    lines = (paths.gold_datasets_dir / "model_versions.csv").read_text().splitlines()
    assert lines == [
        "version,trained_at,training_data_until",
        "1,2026-10-08T16:17:12+00:00,2026-10-05",
        "2,2026-11-20T08:00:00+00:00,2026-11-18",
    ]


def test_latest_model_version_is_last_row(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    append_model_version(paths, trained_at="2026-10-08T16:17:12+00:00", training_data_until="2026-10-05")
    append_model_version(paths, trained_at="2026-11-20T08:00:00+00:00", training_data_until="2026-11-18")

    assert read_latest_model_version(paths) == {
        "version": 2,
        "trained_at": "2026-11-20T08:00:00+00:00",
        "training_data_until": "2026-11-18",
    }


def test_latest_model_version_is_none_without_registry(tmp_path: Path) -> None:
    assert read_latest_model_version(_paths(tmp_path)) is None
