from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from ni_ai_pipeline.paths import PathConfig


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
STUTTGART_LAT = 48.7735
STUTTGART_LON = 9.17868
HOURLY_VARS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "precipitation_probability",
    "precipitation",
    "rain",
    "showers",
    "weather_code",
]


def _validate_requested_hour(requested_hour: int | None) -> None:
    if requested_hour is None:
        return
    if requested_hour < 0 or requested_hour > 23:
        raise ValueError(f"requested_hour must be between 0 and 23, got {requested_hour}")


def _select_target_hour(*, requested_hour: int | None, timezone: str) -> tuple[datetime, int]:
    tz = ZoneInfo(timezone)
    now_local = datetime.now(tz)
    hour = now_local.hour if requested_hour is None else requested_hour
    target_local = now_local.replace(hour=hour, minute=0, second=0, microsecond=0)
    return target_local, hour


def _fetch_open_meteo_hourly_for_range(
    *,
    start_day_local: datetime,
    end_day_local: datetime,
    timezone: str,
) -> pd.DataFrame:
    start_day_str = start_day_local.strftime("%Y-%m-%d")
    end_day_str = end_day_local.strftime("%Y-%m-%d")
    params = {
        "latitude": STUTTGART_LAT,
        "longitude": STUTTGART_LON,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": timezone,
        "start_date": start_day_str,
        "end_date": end_day_str,
    }

    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise RuntimeError("Open-Meteo response missing 'hourly.time'")

    out = pd.DataFrame({"weather_time_local": hourly["time"]})
    for var_name in HOURLY_VARS:
        out[var_name] = hourly.get(var_name)

    out["weather_time_local"] = pd.to_datetime(out["weather_time_local"], errors="coerce")
    out = out.dropna(subset=["weather_time_local"]).sort_values("weather_time_local")
    return out


def pull_stuttgart_weather_hourly(
    paths: PathConfig,
    *,
    requested_hour: int | None = None,
    timezone: str = "Europe/Berlin",
    table_path: Path | None = None,
    upsert: bool = True,
) -> Path:
    """Pull Stuttgart weather for the entire next local calendar day.

    All hourly rows for tomorrow (00:00-23:00 local time) are upserted.
    """

    _validate_requested_hour(requested_hour)

    table_path = table_path or (paths.gold_datasets_dir / "stuttgart_weather.csv")
    target_local, _ = _select_target_hour(requested_hour=requested_hour, timezone=timezone)

    tomorrow_local = target_local + pd.Timedelta(days=1)
    start_day = datetime.combine(tomorrow_local.date(), datetime.min.time(), tzinfo=target_local.tzinfo)
    end_day = start_day
    hourly_df = _fetch_open_meteo_hourly_for_range(
        start_day_local=start_day,
        end_day_local=end_day,
        timezone=timezone,
    )

    if hourly_df.empty:
        raise RuntimeError(
            f"No hourly weather rows found for next day {tomorrow_local.strftime('%Y-%m-%d')}"
        )

    selected = hourly_df.sort_values("weather_time_local").copy()

    selected.insert(0, "site_id", paths.site_id or "stuttgart")
    selected["timezone"] = timezone
    selected["selected_hour"] = selected["weather_time_local"].dt.hour
    selected["source"] = "open-meteo"
    selected["created_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()

    table_path.parent.mkdir(parents=True, exist_ok=True)
    if table_path.exists():
        existing = pd.read_csv(table_path)
        existing["weather_time_local"] = pd.to_datetime(existing.get("weather_time_local"), errors="coerce")
        selected["weather_time_local"] = pd.to_datetime(selected["weather_time_local"], errors="coerce")
        table = pd.concat([existing, selected], ignore_index=True)
    else:
        table = selected

    if upsert:
        table = table.drop_duplicates(subset=["site_id", "weather_time_local"], keep="last")

    table = table.sort_values(["weather_time_local", "site_id"])

    out = table.copy()
    out["weather_time_local"] = pd.to_datetime(out["weather_time_local"], errors="coerce").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    out.to_csv(table_path, index=False)

    print(f"Wrote: {table_path}")
    return table_path
