from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from ni_ai_pipeline.paths import PathConfig


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def _pick_timestamp_column(df: pd.DataFrame) -> str:
    if "DATUM" in df.columns:
        return "DATUM"
    for col in ["MESS_DATUM", "MESS_DATUM_BEGINN", "MESS_DATUM_BEG", "MESS_DATUM_ANFANG"]:
        if col in df.columns:
            return col
    raise KeyError(f"No timestamp column found. Columns: {list(df.columns)}")


def _fix_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df)
    ts_col = _pick_timestamp_column(df)

    if ts_col == "DATUM":
        df["DATUM"] = pd.to_datetime(df["DATUM"], errors="coerce")
    else:
        df["DATUM"] = df[ts_col].astype(str).str.slice(0, 10)
        df["DATUM"] = pd.to_datetime(df["DATUM"], format="%Y%m%d%H", errors="coerce")

    df = df.dropna(subset=["DATUM"]).copy()
    df = df.set_index("DATUM")
    return df


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Placeholder for future explicit renames.
    return df


def _remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    cols_to_drop = [
        "EOR",
        "STATIONS_ID",
        "MESS_DATUM",
        "MESS_DATUM_BEGINN",
        "MESS_DATUM_BEG",
        "MESS_DATUM_ANFANG",
    ]
    # Drop quality-level and previously suffixed duplicate columns that cause
    # repeated merge collisions across category files.
    cols_to_drop.extend([c for c in df.columns if c.startswith("QN_") or c.endswith("_Y")])
    return df.drop(columns=cols_to_drop, errors="ignore")


def _replace_negative_numeric_with_null(
    df: pd.DataFrame, exclude_cols: set[str] | None = None
) -> pd.DataFrame:
    """Replace negative numeric values with nulls while preserving non-numeric fields."""

    # Work on object dtype to avoid StringDtype setitem errors on pandas>=2 when
    # writing numeric/null values back into originally string-typed columns.
    out = df.copy().astype("object")
    excluded = exclude_cols or set()

    # Iterate by position so duplicate column names are handled safely.
    for idx, col in enumerate(out.columns):
        if col in excluded:
            continue

        series = out.iloc[:, idx]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == 0:
            continue

        out.iloc[:, idx] = numeric.mask(numeric < 0)

    return out


def _create_cleans(paths: PathConfig) -> None:
    paths.silver_weather_dir.mkdir(parents=True, exist_ok=True)

    all_files = glob.glob(str(paths.bronze_dwd_dir / "schnarrenberg_dwd*.csv"))
    if not all_files:
        raise FileNotFoundError(
            f"No DWD CSVs found in {paths.bronze_dwd_dir}. "
            "Run scripts/bronze/crawl_dwd.ipynb first or set BRONZE_DWD_DIR/DATA_BRONZE."
        )

    all_files = [f for f in all_files if "regen_rollups" not in Path(f).name.lower()]

    for file in all_files:
        try:
            df = pd.read_csv(file, delimiter=";", header=0, dtype=str)
            if len(df.columns) == 1 and "," in str(df.columns[0]):
                df = pd.read_csv(file, delimiter=",", header=0)
        except Exception as exc:
            print(f"Skipping unreadable file: {file} ({exc})")
            continue

        df_std = _standardize_columns(df)
        has_ts = any(
            c in df_std.columns
            for c in [
                "MESS_DATUM",
                "MESS_DATUM_BEGINN",
                "MESS_DATUM_BEG",
                "MESS_DATUM_ANFANG",
                "DATUM",
            ]
        )
        if not has_ts:
            print(
                f"Skipping non-DWD-raw file (no timestamp col): {file} columns={list(df_std.columns)}"
            )
            continue

        df_fixed = _fix_dates(df)
        df_fixed = _rename_columns(df_fixed)
        df_fixed = _replace_negative_numeric_with_null(df_fixed)

        out_file = paths.silver_weather_dir / f"clean_{Path(file).name}"
        df_fixed.to_csv(out_file)


def _create_master(paths: PathConfig) -> pd.DataFrame:
    all_files = sorted(glob.glob(str(paths.silver_weather_dir / "clean_schnarrenberg_dwd*.csv")))
    if not all_files:
        raise FileNotFoundError(f"No cleaned DWD files found in: {paths.silver_weather_dir}")

    max_dates: list[pd.Timestamp] = []
    prepared: list[pd.DataFrame] = []

    for file in all_files:
        df_new = pd.read_csv(file, delimiter=",", header=0)
        df_new = _remove_duplicate_columns(df_new)
        df_new["DATUM"] = pd.to_datetime(df_new["DATUM"], errors="coerce")
        df_new = df_new.dropna(subset=["DATUM"]).copy()
        if not df_new.empty:
            max_dates.append(df_new["DATUM"].max())
        df_new = df_new.set_index("DATUM")

        # Ensure one row per timestamp to avoid cartesian merge explosions.
        df_new.index = pd.to_datetime(df_new.index, errors="coerce")
        df_new = df_new[~df_new.index.isna()].copy()
        df_new = df_new.sort_index()
        df_new = df_new[~df_new.index.duplicated(keep="last")].copy()

        prepared.append(df_new)

    if not max_dates:
        raise ValueError("No valid DATUM values found in cleaned DWD files.")

    dynamic_end = max(max_dates).floor("h")
    date_range = pd.date_range(start="2023-01-01", end=dynamic_end, freq="h")
    merged = pd.DataFrame(index=date_range)

    for df_new in prepared:
        # Keep first-seen column names and skip repeated duplicates from other files.
        overlap = [c for c in df_new.columns if c in merged.columns]
        if overlap:
            df_new = df_new.drop(columns=overlap, errors="ignore")
        merged = merged.join(df_new, how="left")

    merged = _replace_negative_numeric_with_null(merged)

    return merged


def build_silver_weather(paths: PathConfig) -> Path:
    """Create Silver weather outputs.

    Writes:
    - `clean_*.csv` per Bronze DWD input
    - `clean_wetter_komplett.csv` merged hourly table
    """

    _create_cleans(paths)
    merged = _create_master(paths)
    merged = merged.dropna(how="all")

    out_path = paths.silver_weather_dir / "clean_wetter_komplett.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path)

    print(f"Wrote: {out_path}")
    return out_path
