from __future__ import annotations

from pathlib import Path

import pandas as pd

from ni_ai_pipeline.paths import PathConfig


def build_data_full(paths: PathConfig, *, start_date: str = "2024-01-01") -> Path:
    """Build Gold hourly dataset `data_full.csv`.

    Merges:
    - Silver weather (`clean_wetter_komplett.csv`)
    - Silver measurements (`messungen_komplett.csv`)
    - Bronze LUBW latest dump (`lubw_download_latest.csv`)
    """

    paths.gold_datasets_dir.mkdir(parents=True, exist_ok=True)

    weather_file = paths.silver_weather_dir / "clean_wetter_komplett.csv"
    if not weather_file.exists():
        raise FileNotFoundError(
            f"Missing {weather_file}. Run the Silver weather step first (clean_dwd_daten port)."
        )

    messungen_file = paths.silver_messungen_dir / "messungen_komplett.csv"
    if not messungen_file.exists():
        raise FileNotFoundError(
            f"Missing {messungen_file}. Run the Silver measurements step first (create_messungen_complete port)."
        )

    df_wetter = pd.read_csv(weather_file, header=0, index_col=0, parse_dates=True)
    df_messungen = pd.read_csv(messungen_file, header=0)
    df_messungen["datum"] = pd.to_datetime(df_messungen["datum"], errors="coerce")
    df_messungen = df_messungen.dropna(subset=["datum"]).set_index("datum")

    df_komplett = pd.merge(df_wetter, df_messungen, how="left", left_index=True, right_index=True)
    df_komplett.index.rename("zeit", inplace=True)

    df_komplett = df_komplett[df_komplett.index >= start_date]

    lubw_file = paths.bronze_lubw_dir / "lubw_download_latest.csv"
    if not lubw_file.exists():
        raise FileNotFoundError(
            f"Missing {lubw_file}. Run bronze LUBW ingestion first (or keep the file there)."
        )

    df_lubw_raw = pd.read_csv(lubw_file, sep=";", parse_dates=["Datum"], dayfirst=True)

    water_col = None
    for candidate in ["Gewaesser", "Gewässer"]:
        if candidate in df_lubw_raw.columns:
            water_col = candidate
            break
    if water_col is None:
        raise KeyError(
            f"Missing water body column in LUBW file. Columns: {list(df_lubw_raw.columns)}"
        )

    df_lubw = df_lubw_raw[["Messstation", water_col, "Parameter", "Datum", "Tagesmittelwert"]].copy()
    if water_col != "Gewaesser":
        df_lubw.rename(columns={water_col: "Gewaesser"}, inplace=True)

    df_lubw["Tagesmittelwert"] = df_lubw["Tagesmittelwert"].astype(str).str.replace(",", ".", regex=False)
    df_lubw["Tagesmittelwert"] = pd.to_numeric(df_lubw["Tagesmittelwert"], errors="coerce")

    df_lubw["Kuerzel"] = (
        df_lubw["Messstation"].astype(str).str[:2].str.capitalize()
        + "_"
        + df_lubw["Gewaesser"].astype(str).str[:2].str.capitalize()
        + "_"
        + df_lubw["Parameter"]
        .astype(str)
        .str.replace("bei .*", "", regex=True)
        .str.replace(" ", "", regex=False)
        .str.replace("ä", "ae", regex=False)
        .str.replace("ö", "oe", regex=False)
        .str.replace("ü", "ue", regex=False)
        .str.replace("ß", "ss", regex=False)
    )

    df_lubw_pivot = df_lubw.pivot(index="Datum", columns="Kuerzel", values="Tagesmittelwert")
    df_lubw_hourly = df_lubw_pivot.resample("1h").ffill()

    df_komplett = df_komplett.merge(df_lubw_hourly, left_index=True, right_index=True)

    df_komplett.columns = [str(col).strip() for col in df_komplett.columns]
    df_komplett.index.name = "zeit"

    out_path = paths.gold_datasets_dir / "data_full.csv"
    df_komplett.to_csv(out_path)

    print(f"Wrote: {out_path} (rows={len(df_komplett)} cols={len(df_komplett.columns)})")
    return out_path
