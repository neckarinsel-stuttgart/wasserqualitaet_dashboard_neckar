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


def _format_midnight_timestamp(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise RuntimeError(f"Could not parse date value: {value!r}")
    return pd.Timestamp(ts).strftime("%Y-%m-%d 00:00:00")


def _standardize_prediction_date_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in ["feature_date", "prediction_date"]:
        if col in out.columns:
            parsed = pd.to_datetime(out[col], errors="coerce")
            out[col] = parsed.dt.strftime("%Y-%m-%d 00:00:00").where(parsed.notna(), out[col])
    return out


def _normalize_date_col(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    return out


def _normalize_prediction_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "target" in out.columns:
        out["target"] = out["target"].astype(str).str.strip().str.lower().replace({"pos_neg": "ecoli"})
    return out


def _coerce_prediction_value_to_bool(value: object, *, threshold: float = 0.5) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "t", "yes", "y", "1"}:
            return True
        if token in {"false", "f", "no", "n", "0"}:
            return False

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise RuntimeError(f"Could not convert prediction value to bool: {value!r}")
    return bool(float(numeric) >= float(threshold))


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

    task = str(metadata.get("task", "")).strip().lower()
    target_name = str(metadata.get("target", "ecoli")).strip().lower()
    output_target = "ecoli" if target_name == "pos_neg" else target_name
    if task and task != "binary_classification":
        if target_name != "ecoli":
            raise RuntimeError(
                "Daily predictions require binary_classification for non-ecoli targets; "
                f"got task={task!r}, target={target_name!r} in {model_metadata_path}."
            )
    if output_target not in {"ecoli"}:
        raise RuntimeError(
            "Daily predictions require target='pos_neg' or target='ecoli'; "
            f"got target={target_name or '<missing>'!r} in {model_metadata_path}."
        )

    latest = _load_latest_feature_row(features_path, site_id=paths.site_id)
    predictions = predict_with_saved_model(model, latest, metadata=metadata)

    if len(predictions) != 1:
        raise RuntimeError(f"Expected a single prediction row, got {len(predictions)}")

    feature_date_ts = pd.to_datetime(latest["date"].iloc[0], errors="coerce")
    if pd.isna(feature_date_ts):
        raise RuntimeError("Could not parse latest feature date.")

    prediction_date_ts = pd.Timestamp(feature_date_ts).normalize() + pd.Timedelta(days=horizon_days)

    best_model = str(metadata.get("best_model", "unknown"))
    row_site_id = str(latest.get("site_id", pd.Series([paths.site_id])).iloc[0])

    raw_prediction = predictions[0]
    if target_name == "ecoli":
        ecoli_threshold = float(metadata.get("ecoli_threshold", 500.0))
        raw_numeric = pd.to_numeric(pd.Series([raw_prediction]), errors="coerce").iloc[0]
        if pd.isna(raw_numeric):
            raise RuntimeError(f"Could not convert ecoli prediction to numeric: {raw_prediction!r}")
        prediction_bool = bool(float(raw_numeric) <= ecoli_threshold)
    else:
        if isinstance(raw_prediction, (bool, np.bool_)):
            prediction_bool = bool(raw_prediction)
        else:
            prediction_bool = bool(pd.to_numeric(pd.Series([raw_prediction]), errors="coerce").iloc[0] >= 0.5)

    record = pd.DataFrame(
        [
            {
                "site_id": row_site_id,
                "target": output_target,
                "feature_date": _format_midnight_timestamp(feature_date_ts),
                "prediction_date": _format_midnight_timestamp(prediction_date_ts),
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
        existing = _normalize_prediction_target_labels(pd.read_csv(predictions_path))
        if "prediction" in existing.columns:
            existing["prediction"] = existing["prediction"].apply(_coerce_prediction_value_to_bool)
        table = pd.concat([existing, record], ignore_index=True)
    else:
        table = record

    if "prediction" in table.columns:
        table["prediction"] = table["prediction"].apply(_coerce_prediction_value_to_bool)

    if upsert:
        key_cols = ["site_id", "target", "prediction_date"]
        table = table.drop_duplicates(subset=key_cols, keep="last")

    if "prediction_date" in table.columns:
        table = table.sort_values(["prediction_date", "site_id", "target"])

    table = _standardize_prediction_date_columns(table)

    table.to_csv(predictions_path, index=False)

    print(f"Wrote: {predictions_path}")
    return predictions_path
