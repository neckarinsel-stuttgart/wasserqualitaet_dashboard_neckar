from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from ni_ai_pipeline.paths import PathConfig
from ni_ai_pipeline.winter_break import winter_break_config_path, winter_break_status


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


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        # first and last day of the break are inclusive
        (date(2026, 11, 1), {"active": True, "start": "2026-11-01", "end": "2027-04-30"}),
        (date(2026, 11, 13), {"active": True, "start": "2026-11-01", "end": "2027-04-30"}),
        (date(2027, 1, 1), {"active": True, "start": "2026-11-01", "end": "2027-04-30"}),
        (date(2027, 4, 30), {"active": True, "start": "2026-11-01", "end": "2027-04-30"}),
        # season: show the next upcoming break
        (date(2027, 5, 1), {"active": False, "start": "2027-11-01", "end": "2028-04-30"}),
        (date(2026, 10, 31), {"active": False, "start": "2026-11-01", "end": "2027-04-30"}),
    ],
)
def test_default_period_without_config(tmp_path: Path, today: date, expected: dict) -> None:
    assert winter_break_status(_paths(tmp_path), today=today) == expected


def test_period_is_read_from_config(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = winter_break_config_path(paths)
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"start": "10-15", "end": "03-31"}), encoding="utf-8")

    assert config == tmp_path / "data" / "config" / "winter_break.json"
    assert winter_break_status(paths, today=date(2026, 10, 20)) == {
        "active": True,
        "start": "2026-10-15",
        "end": "2027-03-31",
    }
    assert winter_break_status(paths, today=date(2027, 4, 1)) == {
        "active": False,
        "start": "2027-10-15",
        "end": "2028-03-31",
    }
