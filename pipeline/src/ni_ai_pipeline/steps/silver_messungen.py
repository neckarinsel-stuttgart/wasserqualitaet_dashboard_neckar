from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ni_ai_pipeline.paths import PathConfig


_GERMAN_MONTH_MAP = {
    # common German abbreviations -> English abbreviations understood by strptime
    "JAN": "Jan",
    "FEB": "Feb",
    "MÄR": "Mar",
    "MAER": "Mar",
    "MRZ": "Mar",
    "APR": "Apr",
    "MAI": "May",
    "JUN": "Jun",
    "JUL": "Jul",
    "AUG": "Aug",
    "SEP": "Sep",
    "OKT": "Oct",
    "NOV": "Nov",
    "DEZ": "Dec",
}


def _parse_german_short_date(date_str: str) -> pd.Timestamp:
    """Parse strings like '01. Jan 24' or '01. Mär 24' into a Timestamp."""
    raw = " ".join(date_str.strip().split()[:3])
    parts = raw.split()
    if len(parts) != 3:
        return pd.NaT

    day_part, month_part, year_part = parts
    month_key = (
        month_part.replace(".", "")
        .replace("ä", "ä")
        .replace("Ä", "Ä")
        .strip()
        .upper()
    )

    month_norm = _GERMAN_MONTH_MAP.get(month_key, None)
    if month_norm is None:
        # try umlaut stripped
        month_key2 = (
            month_key.replace("Ä", "AE")
            .replace("ä", "ae")
            .replace("Ö", "OE")
            .replace("ö", "oe")
            .replace("Ü", "UE")
            .replace("ü", "ue")
        ).upper()
        month_norm = _GERMAN_MONTH_MAP.get(month_key2, None)

    if month_norm is None:
        return pd.NaT

    normalized = f"{day_part} {month_norm} {year_part}"
    try:
        dt = datetime.strptime(normalized, "%d. %b %y")
        return pd.Timestamp(dt)
    except Exception:
        return pd.NaT


def _clean_value(val: object) -> float:
    if pd.isna(val) or val == "":
        return float("nan")
    val_str = str(val).replace(">", "").replace(",", ".")
    try:
        return float(val_str)
    except Exception:
        return float("nan")


def _norm_token(value: object) -> str:
    return str(value).strip().lower().replace("_", "")


def _pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {_norm_token(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_token(cand)
        if key in normalized:
            return normalized[key]
    return None


def _build_2024_from_tidy(df_raw: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_existing_column(df_raw, ["DATUM", "datum", "date"])
    ecoli_col = _pick_existing_column(df_raw, ["ecoli", "e_coli", "e. coli", "e.coli"])
    entro_col = _pick_existing_column(df_raw, ["entro", "enterokokken", "enterococcus"])

    if date_col is None or ecoli_col is None or entro_col is None:
        raise KeyError(
            "Could not detect tidy measurement columns for 2024. "
            f"Columns found: {list(df_raw.columns)}"
        )

    out = pd.DataFrame(
        {
            "datum": pd.to_datetime(df_raw[date_col], errors="coerce", dayfirst=False),
            "ecoli": df_raw[ecoli_col].map(_clean_value),
            "entro": df_raw[entro_col].map(_clean_value),
        }
    )
    return out.dropna(subset=["datum"])


def _build_2024_from_wide(df_raw: pd.DataFrame) -> pd.DataFrame:
    idx_lookup = {_norm_token(i): i for i in df_raw.index}

    ecoli_idx = idx_lookup.get(_norm_token("E. Coli")) or idx_lookup.get(_norm_token("E.Coli"))
    entro_idx = idx_lookup.get(_norm_token("Enterokokken"))

    if ecoli_idx is None or entro_idx is None:
        raise KeyError(
            "Could not detect wide measurement rows for 2024. "
            f"Index labels found: {list(df_raw.index)}"
        )

    dates_2024: list[pd.Timestamp] = []
    for col in df_raw.columns:
        dates_2024.append(_parse_german_short_date(str(col)))

    ecoli_2024 = df_raw.loc[ecoli_idx].values
    entro_2024 = df_raw.loc[entro_idx].values

    out = pd.DataFrame(
        {
            "datum": dates_2024,
            "ecoli": [_clean_value(v) for v in ecoli_2024],
            "entro": [_clean_value(v) for v in entro_2024],
        }
    )
    return out.dropna(subset=["datum"])


def build_messungen_komplett(paths: PathConfig) -> pd.DataFrame:
    """Create Silver measurement table from the 2024/2025 Bronze measurement inputs."""

    paths.silver_messungen_dir.mkdir(parents=True, exist_ok=True)

    required = ["messungen_2024.csv", "messungen_2025.csv"]
    missing = [f for f in required if not (paths.bronze_messungen_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {paths.bronze_messungen_dir}. "
            "Put files under data/bronze/messungen (or set BRONZE_MESSUNGEN_DIR/DATA_BRONZE)."
        )

    df_2024_raw = pd.read_csv(
        paths.bronze_messungen_dir / "messungen_2024.csv",
        sep=None,
        engine="python",
    )

    # 2024 arrives in two known shapes:
    # 1) legacy wide matrix (E. Coli/Enterokokken as index rows),
    # 2) tidy rows with datum/ecoli/entro columns.
    tidy_date_col = _pick_existing_column(df_2024_raw, ["DATUM", "datum", "date"])
    if tidy_date_col is not None:
        df_2024 = _build_2024_from_tidy(df_2024_raw)
    else:
        df_2024_wide = pd.read_csv(
            paths.bronze_messungen_dir / "messungen_2024.csv",
            sep=";",
            index_col=0,
        )
        df_2024 = _build_2024_from_wide(df_2024_wide)

    df_2025_raw = pd.read_csv(paths.bronze_messungen_dir / "messungen_2025.csv", sep=";")
    if "DATUM" not in df_2025_raw.columns:
        raise KeyError(f"Missing 'DATUM' in 2025 measurements. Columns: {list(df_2025_raw.columns)}")

    df_2025 = df_2025_raw.copy()
    df_2025["datum"] = pd.to_datetime(df_2025["DATUM"], format="%d.%m.%Y", errors="coerce")

    for col in ["ecoli", "entro"]:
        if col in df_2025.columns:
            df_2025[col] = df_2025[col].map(_clean_value)

    df_2025 = df_2025[["datum", "ecoli", "entro"]]

    df_combined = pd.concat([df_2024, df_2025], ignore_index=True)
    df_combined = df_combined.dropna(subset=["datum"])
    df_combined = df_combined.sort_values("datum").reset_index(drop=True)

    output_path = paths.silver_messungen_dir / "messungen_komplett.csv"
    df_combined.to_csv(output_path, index=False)

    print(f"Wrote: {output_path} (rows={len(df_combined)})")
    return df_combined
