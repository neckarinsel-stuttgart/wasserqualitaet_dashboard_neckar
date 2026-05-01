from __future__ import annotations

from pathlib import Path

import pandas as pd

from ni_ai_pipeline.paths import PathConfig


def build_masterdata_daily(paths: PathConfig, *, start_date: str = "2024-01-01") -> Path:
    """Aggregate `data_full.csv` (hourly) to daily masterdata.csv."""

    paths.gold_datasets_dir.mkdir(parents=True, exist_ok=True)

    data_full_file = paths.gold_datasets_dir / "data_full.csv"
    if not data_full_file.exists():
        raise FileNotFoundError(
            f"Missing {data_full_file}. Run the Gold data_full step first to generate it."
        )

    df_full = pd.read_csv(data_full_file, index_col=0, parse_dates=True)
    df_full = df_full[df_full.index >= start_date]
    df_full = df_full.apply(pd.to_numeric, errors="coerce")

    def sample_at_hour(df: pd.DataFrame, hour: int) -> pd.DataFrame:
        return df[df.index.hour == hour].resample("D").first()

    daily_mean = df_full.resample("D").mean().add_suffix("_mean")
    daily_min = df_full.resample("D").min().add_suffix("_min")
    daily_max = df_full.resample("D").max().add_suffix("_max")

    at_06 = sample_at_hour(df_full, 6).add_suffix("_06")
    at_12 = sample_at_hour(df_full, 12).add_suffix("_12")
    at_18 = sample_at_hour(df_full, 18).add_suffix("_18")
    at_00 = sample_at_hour(df_full, 0).add_suffix("_00")

    df_daily = pd.concat([daily_mean, daily_min, daily_max, at_06, at_12, at_18, at_00], axis=1)
    df_daily.sort_index(inplace=True)

    # Keep same behavior as the notebook (redundant but safe).
    df_daily = df_daily.groupby(df_daily.index.date, group_keys=False).apply(lambda g: g.bfill().ffill())

    out_path = paths.gold_datasets_dir / "masterdata.csv"
    df_daily.to_csv(out_path)

    print(f"Wrote: {out_path} (rows={len(df_daily)} cols={len(df_daily.columns)})")
    return out_path
