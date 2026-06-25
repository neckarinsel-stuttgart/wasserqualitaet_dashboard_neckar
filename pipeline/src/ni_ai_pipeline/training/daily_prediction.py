from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ni_ai_pipeline.paths import PathConfig
from ni_ai_pipeline.training.ecoli_predictability import (
    load_model_metadata,
    load_saved_model,
    predict_with_saved_model,
)


def _normalize_date_col(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    return out


def _load_latest_feature_row(
    features_path: Path,
    *,
    site_id: str | None,
    date_col: str = "date",
) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Features dataset not found: {features_path}")

    df = pd.read_csv(features_path)
    if date_col not in df.columns:
        raise KeyError(f"Missing date column '{date_col}' in {features_path}")

    if site_id is not None and "site_id" in df.columns:
        df = df.loc[df["site_id"].astype(str) == str(site_id)].copy()

    df = _normalize_date_col(df, date_col=date_col)
    if df.empty:
        raise RuntimeError("No rows available in features dataset after filtering.")

    latest_date = df[date_col].max()
    latest = df.loc[df[date_col] == latest_date].tail(1).copy()
    return latest


def predict_latest_and_upsert(
    paths: PathConfig,
    *,
    features_path: Path | None = None,
    model_path: Path | None = None,
    model_metadata_path: Path | None = None,
    predictions_path: Path | None = None,
    horizon_days: int = 1,
    upsert: bool = True,
) -> Path:
    """Predict from latest daily features and append/upsert into a gold predictions CSV table."""

    if horizon_days < 0:
        raise ValueError(f"horizon_days must be >= 0, got {horizon_days}")

    features_path = features_path or (paths.gold_datasets_dir / "gold_daily_features.csv")
    model_path = model_path or (paths.gold_datasets_dir / "ecoli_model.pkl")
    model_metadata_path = model_metadata_path or (paths.gold_datasets_dir / "ecoli_model_metadata.json")
    predictions_path = predictions_path or (paths.gold_datasets_dir / "predictions.csv")

    model = load_saved_model(model_path)
    metadata = load_model_metadata(model_metadata_path)

    latest = _load_latest_feature_row(features_path, site_id=paths.site_id)
    predictions = predict_with_saved_model(model, latest, metadata=metadata)

    if len(predictions) != 1:
        raise RuntimeError(f"Expected a single prediction row, got {len(predictions)}")

    feature_date_ts = pd.to_datetime(latest["date"].iloc[0], errors="coerce")
    if pd.isna(feature_date_ts):
        raise RuntimeError("Could not parse latest feature date.")

    prediction_date_ts = pd.Timestamp(feature_date_ts).normalize() + pd.Timedelta(days=horizon_days)

    target_name = str(metadata.get("target", "ecoli"))
    best_model = str(metadata.get("best_model", "unknown"))
    row_site_id = str(latest.get("site_id", pd.Series([paths.site_id])).iloc[0])

    raw_prediction = predictions[0]
    if isinstance(raw_prediction, (bool, np.bool_)):
        prediction_bool = bool(raw_prediction)
    else:
        prediction_bool = bool(pd.to_numeric(pd.Series([raw_prediction]), errors="coerce").iloc[0] >= 0.5)

    record = pd.DataFrame(
        [
            {
                "site_id": row_site_id,
                "target": target_name,
                "feature_date": pd.Timestamp(feature_date_ts).strftime("%Y-%m-%d"),
                "prediction_date": prediction_date_ts.strftime("%Y-%m-%d"),
                "prediction": prediction_bool,
                "model_name": best_model,
                "model_path": str(model_path),
                "model_metadata_path": str(model_metadata_path),
                "features_path": str(features_path),
                "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
        ]
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        existing = pd.read_csv(predictions_path)
        table = pd.concat([existing, record], ignore_index=True)
    else:
        table = record

    if upsert:
        key_cols = ["site_id", "target", "prediction_date"]
        table = table.drop_duplicates(subset=key_cols, keep="last")

    if "prediction_date" in table.columns:
        table = table.sort_values(["prediction_date", "site_id", "target"])

    table.to_csv(predictions_path, index=False)

    print(f"Wrote: {predictions_path}")
    return predictions_path
