from __future__ import annotations

import pandas as pd

from ni_ai_pipeline.paths import PathConfig


def _clean_value(val: object) -> float:
    if pd.isna(val) or val == "":
        return float("nan")
    val_str = str(val).replace(">", "").replace(",", ".")
    try:
        return float(val_str)
    except Exception:
        return float("nan")


def _normalize_column_name(col: object) -> str:
    # Strip UTF-8 BOM and whitespace to handle files saved with BOM headers.
    return str(col).replace("\ufeff", "").strip()


def _parse_measurement_dates(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    # ISO-like timestamps (e.g. 2024-05-13 08:00:00)
    iso_mask = raw.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(raw.loc[iso_mask], errors="coerce", dayfirst=False)

    # German-style dates (e.g. 06.05.2025)
    non_iso_mask = ~iso_mask
    if non_iso_mask.any():
        parsed.loc[non_iso_mask] = pd.to_datetime(raw.loc[non_iso_mask], errors="coerce", dayfirst=True)

    return parsed


def _read_tidy_measurements(csv_path: pd.io.common.FilePath) -> pd.DataFrame:
    """Read row-wise measurements with columns DATUM/ecoli/entro."""
    df_raw = pd.read_csv(csv_path, sep=None, engine="python")
    df_raw.columns = [_normalize_column_name(c) for c in df_raw.columns]

    columns_lower = {str(col).strip().lower(): col for col in df_raw.columns}
    required_cols = ["datum", "ecoli", "entro"]
    missing_cols = [col for col in required_cols if col not in columns_lower]
    if missing_cols:
        raise KeyError(
            f"Missing columns {missing_cols} in {csv_path}. "
            f"Expected DATUM/ecoli/entro style input. Found: {list(df_raw.columns)}"
        )

    datum_col = columns_lower["datum"]
    ecoli_col = columns_lower["ecoli"]
    entro_col = columns_lower["entro"]

    out = pd.DataFrame(
        {
            "datum": _parse_measurement_dates(df_raw[datum_col]),
            "ecoli": df_raw[ecoli_col].map(_clean_value),
            "entro": df_raw[entro_col].map(_clean_value),
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

    df_2024 = _read_tidy_measurements(paths.bronze_messungen_dir / "messungen_2024.csv")
    df_2025 = _read_tidy_measurements(paths.bronze_messungen_dir / "messungen_2025.csv")

    df_combined = pd.concat([df_2024, df_2025], ignore_index=True)
    df_combined = df_combined.dropna(subset=["datum"])
    df_combined = df_combined.sort_values("datum").reset_index(drop=True)

    output_path = paths.silver_messungen_dir / "messungen_komplett.csv"
    df_combined.to_csv(output_path, index=False)

    print(f"Wrote: {output_path} (rows={len(df_combined)})")
    return df_combined
