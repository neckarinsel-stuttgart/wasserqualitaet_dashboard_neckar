from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ni_ai_pipeline.paths import PathConfig


def _impute_small_gaps_mean_of_neighbors(
    df: pd.DataFrame,
    *,
    max_gap: int = 2,
    min_non_null_neighbors: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Impute only short internal gaps (length <= max_gap).

    For a run of NaNs bounded by valid values on both sides, fill the NaNs
    with the mean of (previous_value, next_value).

    This avoids smoothing long gaps or extrapolating at the ends.
    """

    if df.empty:
        return df

    out = df.copy()
    total_filled = 0

    for col in out.columns:
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue

        # Force a writable mask array; some pandas/numpy combinations return
        # read-only views from to_numpy().
        is_na = s.isna().to_numpy(copy=True)
        if not is_na.any():
            continue

        values = s.to_numpy(copy=True)
        n = len(values)
        i = 0
        filled_col = 0

        while i < n:
            if not is_na[i]:
                i += 1
                continue

            j = i
            while j < n and is_na[j]:
                j += 1
            run_len = j - i

            if run_len <= max_gap:
                left_idx = i - 1
                right_idx = j

                neighbors = 0
                if left_idx >= 0 and pd.notna(values[left_idx]):
                    neighbors += 1
                if right_idx < n and pd.notna(values[right_idx]):
                    neighbors += 1

                if neighbors >= min_non_null_neighbors and left_idx >= 0 and right_idx < n:
                    left_val = values[left_idx]
                    right_val = values[right_idx]
                    fill_val = (left_val + right_val) / 2
                    values[i:j] = fill_val
                    is_na[i:j] = False
                    filled_col += run_len

            i = j

        if filled_col:
            out[col] = values
            total_filled += filled_col

    if verbose:
        print(
            f"Imputed short gaps (<= {max_gap} day(s)) across all features: filled_cells={total_filled:,}"
        )

    return out


def build_daily_gold_dataset(
    paths: PathConfig,
    *,
    shift_days: int = 1,
) -> tuple[Path, Path, Path]:
    """Create daily-grain Gold outputs.

    Writes:
    - `gold_daily_features.csv`
    - `gold_daily_labels.csv`
    - `gold_daily_dataset.csv`

    Leakage-safe default: label date D uses features derived from weather up to D-1.
    """

    paths.gold_datasets_dir.mkdir(parents=True, exist_ok=True)

    weather_path = paths.silver_weather_dir / "clean_wetter_komplett.csv"
    labels_path = paths.silver_messungen_dir / "messungen_komplett.csv"

    if not weather_path.exists():
        raise FileNotFoundError(f"Missing Silver weather file: {weather_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing Silver labels file: {labels_path}")

    weather = pd.read_csv(weather_path, index_col=0, parse_dates=True, low_memory=False)
    weather.index.name = "timestamp"
    weather = weather.sort_index()

    weather.columns = [str(c).strip() for c in weather.columns]
    weather_numeric = weather.apply(pd.to_numeric, errors="coerce")

    upper_cols = {c: c.upper() for c in weather_numeric.columns}
    drop_cols: list[str] = []
    for col, col_upper in upper_cols.items():
        if col_upper.startswith("QN_"):
            drop_cols.append(col)
        if col_upper in {
            "EOR",
            "STATIONS_ID",
            "MESS_DATUM",
            "MESS_DATUM_BEGINN",
            "MESS_DATUM_BEG",
            "MESS_DATUM_ANFANG",
        }:
            drop_cols.append(col)
        if col_upper.endswith("_Y"):
            drop_cols.append(col)

    drop_cols = sorted({c for c in drop_cols if c in weather_numeric.columns})
    weather_numeric = weather_numeric.drop(columns=drop_cols, errors="ignore")

    daily_mean = weather_numeric.resample("D").mean()
    daily_min = weather_numeric.resample("D").min()
    daily_max = weather_numeric.resample("D").max()

    sum_like_candidates = {"R1", "RS_IN", "SD_SO", "SD_LBERG", "SD_DUETT", "SD_UN_DUETT"}
    sum_cols = [
        c
        for c in weather_numeric.columns
        if c.upper() in sum_like_candidates or c.upper().startswith("R1")
    ]
    daily_sum = (
        weather_numeric[sum_cols].resample("D").sum(min_count=1)
        if sum_cols
        else pd.DataFrame(index=daily_mean.index)
    )

    missing_frac_per_col = weather_numeric.isna().resample("D").mean()
    weather_missing_frac = missing_frac_per_col.mean(axis=1).rename("weather_missing_frac")

    features = pd.DataFrame(index=daily_mean.index)
    features = features.join(daily_mean.add_suffix("_mean"))
    features = features.join(daily_min.add_suffix("_min"))
    features = features.join(daily_max.add_suffix("_max"))
    if not daily_sum.empty:
        features = features.join(daily_sum.add_suffix("_sum"))
    features = features.join(weather_missing_frac)

    features = features.sort_index().asfreq("D")
    features = _impute_small_gaps_mean_of_neighbors(features, max_gap=2, verbose=True)

    preferred = [
        "TT_TU_mean",
        "RF_TU_mean",
        "P_mean",
        "P0_mean",
        "R1_sum",
        "RS_IN_sum",
        "SD_SO_sum",
        "weather_missing_frac",
    ]
    preferred = [c for c in preferred if c in features.columns]

    for col in preferred:
        if col.endswith("_sum"):
            features[f"{col}_3d"] = features[col].rolling(3, min_periods=1).sum()
            features[f"{col}_7d"] = features[col].rolling(7, min_periods=1).sum()
        else:
            features[f"{col}_3d"] = features[col].rolling(3, min_periods=1).mean()
            features[f"{col}_7d"] = features[col].rolling(7, min_periods=1).mean()
        features[f"{col}_lag1"] = features[col].shift(1)
        features[f"{col}_lag7"] = features[col].shift(7)

    features_for_labels = features.shift(shift_days)

    labels_raw = pd.read_csv(labels_path)
    labels_raw.columns = [str(c).strip().lower() for c in labels_raw.columns]
    if "datum" not in labels_raw.columns:
        raise KeyError(f"Expected column 'datum' in labels file. Columns: {list(labels_raw.columns)}")

    labels_raw["date"] = pd.to_datetime(labels_raw["datum"], errors="coerce").dt.normalize()
    labels_raw = labels_raw.dropna(subset=["date"])

    value_cols = [c for c in ["ecoli", "entro"] if c in labels_raw.columns]
    agg_map = {c: "mean" for c in value_cols}
    labels = labels_raw.groupby("date", as_index=False).agg({**agg_map, "datum": "count"})
    labels = labels.rename(columns={"datum": "n_samples"})
    labels.insert(0, "site_id", paths.site_id)
    if "ecoli" in labels.columns:
        ecoli_numeric = pd.to_numeric(labels["ecoli"], errors="coerce")
        labels["pos_neg"] = np.where(ecoli_numeric <= 1000, 1, 0)
        labels.loc[ecoli_numeric.isna(), "pos_neg"] = np.nan

    features_out = features.reset_index().rename(columns={"index": "date", "timestamp": "date"})
    features_out["date"] = pd.to_datetime(features_out["date"]).dt.normalize()
    features_out.insert(0, "site_id", paths.site_id)

    features_for_labels_out = features_for_labels.reset_index().rename(
        columns={"index": "date", "timestamp": "date"}
    )
    features_for_labels_out["date"] = pd.to_datetime(features_for_labels_out["date"]).dt.normalize()
    features_for_labels_out.insert(0, "site_id", paths.site_id)

    final = labels.merge(
        features_for_labels_out,
        on=["site_id", "date"],
        how="left",
        validate="one_to_one",
    )

    features_path = paths.gold_datasets_dir / "gold_daily_features.csv"
    labels_path_out = paths.gold_datasets_dir / "gold_daily_labels.csv"
    final_path = paths.gold_datasets_dir / "gold_daily_dataset.csv"

    features_out.to_csv(features_path, index=False)
    labels.to_csv(labels_path_out, index=False)
    final.to_csv(final_path, index=False)

    print("Wrote:")
    print(f"  {features_path}")
    print(f"  {labels_path_out}")
    print(f"  {final_path}")

    return features_path, labels_path_out, final_path
