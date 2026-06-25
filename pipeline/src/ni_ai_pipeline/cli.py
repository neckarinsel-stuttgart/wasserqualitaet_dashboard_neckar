from __future__ import annotations

import argparse
from pathlib import Path

from ni_ai_pipeline.paths import get_paths
from ni_ai_pipeline.steps.gold_daily_dataset import build_daily_gold_dataset
from ni_ai_pipeline.steps.gold_data_full import build_data_full
from ni_ai_pipeline.steps.gold_masterdata import build_masterdata_daily
from ni_ai_pipeline.steps.gold_stuttgart_weather import pull_stuttgart_weather_hourly
from ni_ai_pipeline.steps.silver_messungen import build_messungen_komplett
from ni_ai_pipeline.steps.silver_weather import build_silver_weather
from ni_ai_pipeline.training.ecoli_predictability import train_ecoli_predictability
from ni_ai_pipeline.training.ecoli_predictability import load_model_metadata
from ni_ai_pipeline.training.daily_prediction import predict_latest_and_upsert


def _needs_classifier_retrain(model_path: Path, model_metadata_path: Path) -> bool:
    if (not model_path.exists()) or (not model_metadata_path.exists()):
        return True

    try:
        metadata = load_model_metadata(model_metadata_path)
    except Exception:
        return True

    return str(metadata.get("task")) != "binary_classification"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ni-ai-pipeline")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("silver-weather", help="Build Silver weather outputs from Bronze DWD CSVs")
    sub.add_parser("silver-messungen", help="Build Silver measurement table messungen_komplett.csv")
    sub.add_parser("gold-data-full", help="Build Gold hourly dataset data_full.csv")
    sub.add_parser("gold-masterdata", help="Aggregate data_full.csv to daily masterdata.csv")
    sub.add_parser("gold-daily-dataset", help="Build daily-grain dataset for modeling")

    stuttgart_weather = sub.add_parser(
        "stuttgart-weather-hourly",
        help="Pull Stuttgart hourly weather and upsert one selected hour into Gold stuttgart_weather.csv",
    )
    stuttgart_weather.add_argument(
        "--hour",
        type=int,
        default=None,
        help="Requested local hour [0-23]. Defaults to current hour in timezone.",
    )
    stuttgart_weather.add_argument(
        "--timezone",
        type=str,
        default="Europe/Berlin",
        help="IANA timezone for hour selection (default: Europe/Berlin)",
    )
    stuttgart_weather.add_argument(
        "--table-path",
        type=Path,
        default=None,
        help="Optional output path for stuttgart_weather table CSV",
    )
    stuttgart_weather.add_argument(
        "--append-only",
        action="store_true",
        help="Append only; do not upsert by site_id/weather_time_local",
    )

    train = sub.add_parser("train-ecoli", help="Evaluate/pick simple models for ecoli predictability")
    train.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to gold_daily_dataset.csv (defaults to configured GOLD_DATASETS_DIR)",
    )
    train.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir for report/artifacts (defaults to configured GOLD_DATASETS_DIR)",
    )
    train.add_argument("--target", type=str, default="ecoli")
    train.add_argument("--test-fraction", type=float, default=0.2)
    train.add_argument("--log-target", action="store_true")
    train.add_argument("--export-ecoli-only", action="store_true")
    train.add_argument(
        "--no-save-model",
        action="store_true",
        help="If set, skip saving the fitted best model artifact and metadata",
    )
    train.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional output path for saved model artifact (pickle)",
    )
    train.add_argument(
        "--model-metadata-path",
        type=Path,
        default=None,
        help="Optional output path for saved model metadata (JSON)",
    )

    train.add_argument(
        "--mlflow",
        action="store_true",
        help="Log params/metrics/artifacts (and model) to MLflow if installed",
    )
    train.add_argument("--mlflow-experiment", type=str, default="ni-ai")
    train.add_argument("--mlflow-run-name", type=str, default=None)
    train.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="Override MLFLOW_TRACKING_URI (e.g. file:./mlruns or http://...) ",
    )
    train.add_argument(
        "--no-mlflow-model",
        action="store_true",
        help="If set, do not log the fitted sklearn pipeline as an MLflow model",
    )

    predict = sub.add_parser(
        "predict-daily",
        help="Predict from latest gold_daily_features row and upsert gold predictions table",
    )
    predict.add_argument(
        "--features-data",
        type=Path,
        default=None,
        help="Path to gold_daily_features.csv (defaults to configured GOLD_DATASETS_DIR)",
    )
    predict.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to saved model pickle (defaults to GOLD_DATASETS_DIR/ecoli_model.pkl)",
    )
    predict.add_argument(
        "--model-metadata-path",
        type=Path,
        default=None,
        help="Path to model metadata JSON (defaults to GOLD_DATASETS_DIR/ecoli_model_metadata.json)",
    )
    predict.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Output predictions table CSV (defaults to GOLD_DATASETS_DIR/predictions.csv)",
    )
    predict.add_argument(
        "--horizon-days",
        type=int,
        default=1,
        help="Prediction horizon in days after feature date",
    )
    predict.add_argument(
        "--append-only",
        action="store_true",
        help="Append rows only; do not upsert existing site/date/target entries",
    )

    sub.add_parser("run-all", help="Run Silver + Gold + (optional) training")
    sub.add_parser(
        "run-daily-midnight",
        help="Run Silver+Gold refresh, then predict latest and upsert predictions table",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = get_paths(load_dotenv=True)

    default_model_path = paths.gold_datasets_dir / "ecoli_model.pkl"
    default_model_metadata_path = paths.gold_datasets_dir / "ecoli_model_metadata.json"

    if args.cmd == "silver-weather":
        build_silver_weather(paths)
        return 0

    if args.cmd == "silver-messungen":
        build_messungen_komplett(paths)
        return 0

    if args.cmd == "gold-data-full":
        build_data_full(paths)
        return 0

    if args.cmd == "gold-masterdata":
        build_masterdata_daily(paths)
        return 0

    if args.cmd == "gold-daily-dataset":
        build_daily_gold_dataset(paths)
        return 0

    if args.cmd == "stuttgart-weather-hourly":
        pull_stuttgart_weather_hourly(
            paths,
            requested_hour=args.hour,
            timezone=args.timezone,
            table_path=args.table_path,
            upsert=(not args.append_only),
        )
        return 0

    if args.cmd == "train-ecoli":
        train_ecoli_predictability(
            paths,
            data_path=args.data,
            out_dir=args.out_dir,
            target=args.target,
            test_fraction=args.test_fraction,
            log_target=args.log_target,
            export_ecoli_only=args.export_ecoli_only,
            save_model=(not args.no_save_model),
            model_path=args.model_path,
            model_metadata_path=args.model_metadata_path,
            mlflow_enabled=args.mlflow,
            mlflow_experiment=args.mlflow_experiment,
            mlflow_run_name=args.mlflow_run_name,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_log_model=(not args.no_mlflow_model),
        )
        return 0

    if args.cmd == "predict-daily":
        # Bootstrap model artifacts on first deployment when default paths are used.
        model_path = args.model_path or default_model_path
        model_metadata_path = args.model_metadata_path or default_model_metadata_path
        if args.model_path is None and args.model_metadata_path is None:
            if _needs_classifier_retrain(model_path, model_metadata_path):
                print(
                    "Model artifacts missing; training once before first prediction "
                    f"({model_path}, {model_metadata_path})."
                )
                train_ecoli_predictability(paths)

        predict_latest_and_upsert(
            paths,
            features_path=args.features_data,
            model_path=model_path,
            model_metadata_path=model_metadata_path,
            predictions_path=args.predictions_path,
            horizon_days=args.horizon_days,
            upsert=(not args.append_only),
        )
        return 0

    if args.cmd == "run-all":
        build_silver_weather(paths)
        build_messungen_komplett(paths)
        build_data_full(paths)
        build_masterdata_daily(paths)
        build_daily_gold_dataset(paths)
        return 0

    if args.cmd == "run-daily-midnight":
        build_silver_weather(paths)
        build_messungen_komplett(paths)
        build_data_full(paths)
        build_masterdata_daily(paths)
        build_daily_gold_dataset(paths)
        if _needs_classifier_retrain(default_model_path, default_model_metadata_path):
            print(
                "Model artifacts missing; training once before first prediction "
                f"({default_model_path}, {default_model_metadata_path})."
            )
            train_ecoli_predictability(paths)
        predict_latest_and_upsert(paths)
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")
