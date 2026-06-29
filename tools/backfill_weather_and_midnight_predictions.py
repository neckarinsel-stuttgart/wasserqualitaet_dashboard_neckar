from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_pythonpath() -> None:
    src = _repo_root() / "pipeline" / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_bootstrap_pythonpath()

from ni_ai_pipeline.paths import get_paths
from ni_ai_pipeline.steps.gold_daily_dataset import build_daily_gold_dataset
from ni_ai_pipeline.steps.gold_data_full import build_data_full
from ni_ai_pipeline.steps.gold_masterdata import build_masterdata_daily
from ni_ai_pipeline.steps.silver_messungen import build_messungen_komplett
from ni_ai_pipeline.steps.silver_weather import build_silver_weather
from ni_ai_pipeline.training.ecoli_predictability import (
    load_model_metadata,
    load_saved_model,
    predict_with_saved_model,
    train_ecoli_predictability,
)


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
STUTTGART_LAT = 48.7735
STUTTGART_LON = 9.17868
HOURLY_VARS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "rain",
    "showers",
    "weather_code",
    "wind_speed_10m"
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill missing Stuttgart hourly weather rows and midnight predictions "
            "for the last N days ending today."
        )
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days ending today to backfill (default: 30)",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="Europe/Berlin",
        help="IANA timezone used for local dates/hours (default: Europe/Berlin)",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=1,
        help="Prediction horizon in days used by midnight prediction flow (default: 1)",
    )
    parser.add_argument(
        "--weather-table-path",
        type=Path,
        default=None,
        help="Optional path to stuttgart_weather.csv",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Optional path to predictions.csv",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Optional path to gold_daily_features.csv",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to ecoli model pickle",
    )
    parser.add_argument(
        "--model-metadata-path",
        type=Path,
        default=None,
        help="Optional path to model metadata JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print missing rows without writing files",
    )
    parser.add_argument(
        "--overwrite-predictions",
        action="store_true",
        help="Regenerate and replace prediction rows in the backfill window instead of skipping existing ones",
    )
    return parser.parse_args()


def _date_window(days: int, timezone: str) -> pd.DatetimeIndex:
    if days <= 0:
        raise ValueError(f"days must be >= 1, got {days}")
    tz = ZoneInfo(timezone)
    today_local = pd.Timestamp(datetime.now(tz).date())
    return pd.date_range(end=today_local, periods=days, freq="D")


def _fetch_open_meteo_hourly_for_day(day_local: pd.Timestamp, timezone: str) -> pd.DataFrame:
    day_str = pd.Timestamp(day_local).strftime("%Y-%m-%d")
    params = {
        "latitude": STUTTGART_LAT,
        "longitude": STUTTGART_LON,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": timezone,
        "start_date": day_str,
        "end_date": day_str,
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


def _backfill_hourly_weather(
    *,
    site_id: str,
    weather_table_path: Path,
    days: int,
    timezone: str,
    dry_run: bool,
) -> tuple[int, int]:
    weather_table_path.parent.mkdir(parents=True, exist_ok=True)

    if weather_table_path.exists():
        existing = pd.read_csv(weather_table_path)
    else:
        existing = pd.DataFrame(columns=["site_id", "weather_time_local"])

    existing["site_id"] = existing.get("site_id", pd.Series(dtype=str)).astype(str)
    existing["weather_time_local"] = pd.to_datetime(existing.get("weather_time_local"), errors="coerce")
    existing = existing.dropna(subset=["weather_time_local"]).copy()

    existing_key = set(zip(existing["site_id"], existing["weather_time_local"]))

    day_range = _date_window(days, timezone)
    day_frames: list[pd.DataFrame] = []
    for day_local in day_range:
        day_df = _fetch_open_meteo_hourly_for_day(day_local=day_local, timezone=timezone)
        day_df.insert(0, "site_id", site_id)
        day_df["timezone"] = timezone
        day_df["selected_hour"] = day_df["weather_time_local"].dt.hour
        day_df["source"] = "open-meteo"
        day_df["created_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        day_frames.append(day_df)

    pulled = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()
    if pulled.empty:
        return 0, len(existing)

    pulled["site_id"] = pulled["site_id"].astype(str)
    pulled["weather_time_local"] = pd.to_datetime(pulled["weather_time_local"], errors="coerce")
    pulled = pulled.dropna(subset=["weather_time_local"]).copy()

    missing_mask = [
        (site, ts) not in existing_key
        for site, ts in zip(pulled["site_id"], pulled["weather_time_local"], strict=False)
    ]
    missing = pulled.loc[missing_mask].copy()

    if missing.empty:
        return 0, len(existing)

    merged = pd.concat([existing, missing], ignore_index=True)
    merged = merged.drop_duplicates(subset=["site_id", "weather_time_local"], keep="last")
    merged = merged.sort_values(["weather_time_local", "site_id"])

    if not dry_run:
        out = merged.copy()
        out["weather_time_local"] = pd.to_datetime(
            out["weather_time_local"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        out.to_csv(weather_table_path, index=False)

    return len(missing), len(merged)


def _normalize_feature_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


def _normalize_prediction_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "target" in out.columns:
        out["target"] = out["target"].astype(str).str.strip().str.lower().replace({"pos_neg": "ecoli"})
    return out


def _prediction_to_bool(value: object, *, threshold: float = 0.5) -> bool:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Could not convert prediction value to bool: {value!r}")
    return bool(float(numeric) >= threshold)


def _ecoli_prediction_to_bool(value: object, *, ecoli_threshold: float = 1000.0) -> bool:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Could not convert ecoli prediction value to numeric: {value!r}")
    return bool(float(numeric) <= float(ecoli_threshold))


def _standardize_prediction_date_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in ["feature_date", "prediction_date"]:
        if col in out.columns:
            parsed = pd.to_datetime(out[col], errors="coerce")
            out[col] = parsed.dt.strftime("%Y-%m-%d 00:00:00").where(parsed.notna(), out[col])
    return out


def _backfill_midnight_predictions(
    *,
    site_id: str,
    features_path: Path,
    model_path: Path,
    model_metadata_path: Path,
    predictions_path: Path,
    days: int,
    timezone: str,
    horizon_days: int,
    dry_run: bool,
    overwrite_predictions: bool,
) -> tuple[int, int, int]:
    if horizon_days < 0:
        raise ValueError(f"horizon_days must be >= 0, got {horizon_days}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features dataset not found: {features_path}")

    model = load_saved_model(model_path)
    metadata = load_model_metadata(model_metadata_path)
    task = str(metadata.get("task", "")).strip().lower()
    target_name = str(metadata.get("target", "ecoli")).strip().lower()
    output_target = "ecoli" if target_name == "pos_neg" else target_name
    if task and task != "binary_classification":
        if target_name != "ecoli":
            raise RuntimeError(
                "Midnight backfill requires binary_classification for non-ecoli targets; "
                f"got task={task!r}, target={target_name!r} in {model_metadata_path}."
            )
    if output_target not in {"ecoli"}:
        raise RuntimeError(
            "Midnight backfill requires target='pos_neg' or target='ecoli'; "
            f"got target={target_name or '<missing>'!r} in {model_metadata_path}."
        )
    ecoli_threshold = float(metadata.get("ecoli_threshold", 1000.0))

    features_df = pd.read_csv(features_path)
    if "date" not in features_df.columns:
        raise KeyError(f"Missing 'date' column in features dataset: {features_path}")

    if "site_id" in features_df.columns:
        features_df = features_df.loc[features_df["site_id"].astype(str) == str(site_id)].copy()
    features_df = _normalize_feature_dates(features_df)

    if predictions_path.exists():
        pred_existing = _normalize_prediction_target_labels(pd.read_csv(predictions_path))
    else:
        pred_existing = pd.DataFrame(columns=["site_id", "target", "prediction_date"])

    pred_existing["site_id"] = pred_existing.get("site_id", pd.Series(dtype=str)).astype(str)
    pred_existing["target"] = pred_existing.get("target", pd.Series(dtype=str)).astype(str)
    pred_existing["prediction_date"] = pd.to_datetime(
        pred_existing.get("prediction_date"), errors="coerce"
    ).dt.normalize()
    pred_existing = pred_existing.dropna(subset=["prediction_date"]).copy()

    existing_key = set(
        zip(
            pred_existing["site_id"],
            pred_existing["target"],
            pred_existing["prediction_date"],
        )
    )

    requested_prediction_dates = _date_window(days, timezone)
    to_add: list[dict[str, object]] = []
    skipped_missing_features = 0

    for prediction_date in requested_prediction_dates:
        key = (str(site_id), output_target, prediction_date)
        if key in existing_key and not overwrite_predictions:
            continue

        feature_date = prediction_date - pd.Timedelta(days=horizon_days)
        feature_rows = features_df.loc[features_df["date"] == feature_date].copy()
        if feature_rows.empty:
            skipped_missing_features += 1
            continue

        latest_feature_row = feature_rows.tail(1).copy()
        pred = predict_with_saved_model(model, latest_feature_row, metadata=metadata)
        if target_name == "ecoli":
            pred_value = _ecoli_prediction_to_bool(np.asarray(pred)[0], ecoli_threshold=ecoli_threshold)
        else:
            pred_value = _prediction_to_bool(np.asarray(pred, dtype=float)[0])

        row_site_id = str(
            latest_feature_row.get("site_id", pd.Series([site_id], index=latest_feature_row.index)).iloc[0]
        )

        to_add.append(
            {
                "site_id": row_site_id,
                "target": output_target,
                "feature_date": feature_date.strftime("%Y-%m-%d 00:00:00"),
                "prediction_date": prediction_date.strftime("%Y-%m-%d 00:00:00"),
                "prediction": pred_value,
                "model_name": str(metadata.get("best_model", "unknown")),
                "model_path": str(model_path),
                "model_metadata_path": str(model_metadata_path),
                "features_path": str(features_path),
                "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
        )

    if not to_add:
        return 0, len(pred_existing), skipped_missing_features

    add_df = pd.DataFrame(to_add)
    if overwrite_predictions:
        replacement_keys = set(
            zip(add_df["site_id"], add_df["target"], pd.to_datetime(add_df["prediction_date"], errors="coerce"))
        )
        pred_existing = pred_existing.loc[
            ~pred_existing.apply(
                lambda row: (row["site_id"], row["target"], row["prediction_date"]) in replacement_keys,
                axis=1,
            )
        ].copy()

    merged = pd.concat([pred_existing, add_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["site_id", "target", "prediction_date"], keep="last")
    merged = merged.sort_values(["prediction_date", "site_id", "target"])
    merged = _standardize_prediction_date_columns(merged)

    if not dry_run:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(predictions_path, index=False)

    return len(add_df), len(merged), skipped_missing_features


def _needs_classifier_retrain(model_path: Path, model_metadata_path: Path) -> bool:
    if (not model_path.exists()) or (not model_metadata_path.exists()):
        return True

    try:
        metadata = load_model_metadata(model_metadata_path)
    except Exception:
        return True

    return str(metadata.get("task")) != "binary_classification"


def main() -> int:
    args = _parse_args()
    paths = get_paths(load_dotenv=True, start=_repo_root())

    weather_table_path = args.weather_table_path or (paths.gold_datasets_dir / "stuttgart_weather.csv")
    features_path = args.features_path or (paths.gold_datasets_dir / "gold_daily_features.csv")
    model_path = args.model_path or (paths.gold_datasets_dir / "ecoli_model.pkl")
    model_metadata_path = args.model_metadata_path or (paths.gold_datasets_dir / "ecoli_model_metadata.json")
    predictions_path = args.predictions_path or (paths.gold_datasets_dir / "predictions.csv")

    if args.model_path is None and args.model_metadata_path is None and _needs_classifier_retrain(
        model_path,
        model_metadata_path,
    ):
        print(
            "Model artifacts missing or outdated; training classifier before backfill "
            f"({model_path}, {model_metadata_path})."
        )
        build_silver_weather(paths)
        build_messungen_komplett(paths)
        build_data_full(paths)
        build_masterdata_daily(paths)
        build_daily_gold_dataset(paths)
        train_ecoli_predictability(paths)

    weather_added, weather_total = _backfill_hourly_weather(
        site_id=paths.site_id,
        weather_table_path=weather_table_path,
        days=args.days,
        timezone=args.timezone,
        dry_run=args.dry_run,
    )

    predictions_added, predictions_total, skipped_no_features = _backfill_midnight_predictions(
        site_id=paths.site_id,
        features_path=features_path,
        model_path=model_path,
        model_metadata_path=model_metadata_path,
        predictions_path=predictions_path,
        days=args.days,
        timezone=args.timezone,
        horizon_days=args.horizon_days,
        dry_run=args.dry_run,
        overwrite_predictions=args.overwrite_predictions,
    )

    mode = "DRY-RUN" if args.dry_run else "WRITE"
    print(f"[{mode}] Weather rows added: {weather_added} (total table rows: {weather_total})")
    print(
        "[{}] Midnight predictions added: {} (total table rows: {}, skipped_no_features: {})".format(
            mode,
            predictions_added,
            predictions_total,
            skipped_no_features,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())