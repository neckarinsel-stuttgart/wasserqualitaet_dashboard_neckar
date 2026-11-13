"""Yearly winter break (no bathing season, no daily updates).

The period is read from data/config/winter_break.json, e.g.
    {"start": "11-01", "end": "04-30"}
Both days are inclusive and the period repeats every year. If the file is missing,
the defaults below are used.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ni_ai_pipeline.paths import PathConfig

DEFAULT_START = "11-01"
DEFAULT_END = "04-30"
TIMEZONE = "Europe/Berlin"


def winter_break_config_path(paths: PathConfig) -> Path:
    return paths.data_gold.parent / "config" / "winter_break.json"


def _parse_month_day(value: str) -> tuple[int, int]:
    month, day = (int(part) for part in str(value).strip().split("-"))
    date(2001, month, day)  # validate (non-leap year)
    return month, day


def load_winter_break_period(paths: PathConfig) -> tuple[tuple[int, int], tuple[int, int]]:
    path = winter_break_config_path(paths)
    config: dict[str, Any] = {}
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
    start = _parse_month_day(config.get("start", DEFAULT_START))
    end = _parse_month_day(config.get("end", DEFAULT_END))
    return start, end


def _window(start_year: int, start: tuple[int, int], end: tuple[int, int]) -> tuple[date, date]:
    first = date(start_year, *start)
    # A period like 11-01..04-30 ends in the following year.
    last = date(start_year + (1 if end < start else 0), *end)
    return first, last


def winter_break_status(paths: PathConfig, *, today: date | None = None) -> dict[str, Any]:
    """Return the current break (active) or the next upcoming one (not active)."""

    if today is None:
        today = pd.Timestamp.now(tz=TIMEZONE).date()
    start, end = load_winter_break_period(paths)

    windows = [_window(year, start, end) for year in (today.year - 1, today.year, today.year + 1)]
    for first, last in windows:
        if first <= today <= last:
            return {"active": True, "start": first.isoformat(), "end": last.isoformat()}

    first, last = min(w for w in windows if w[0] > today)
    return {"active": False, "start": first.isoformat(), "end": last.isoformat()}
