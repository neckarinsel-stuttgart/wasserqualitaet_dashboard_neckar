from __future__ import annotations

import argparse
from pathlib import Path

from ni_ai_pipeline.paths import get_paths
from ni_ai_pipeline.steps.gold_daily_dataset import build_daily_gold_dataset
from ni_ai_pipeline.steps.gold_data_full import build_data_full
from ni_ai_pipeline.steps.gold_masterdata import build_masterdata_daily
from ni_ai_pipeline.steps.silver_messungen import build_messungen_komplett
from ni_ai_pipeline.steps.silver_weather import build_silver_weather
from ni_ai_pipeline.training.ecoli_predictability import train_ecoli_predictability


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ni-ai-pipeline")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("silver-weather", help="Build Silver weather outputs from Bronze DWD CSVs")
    sub.add_parser("silver-messungen", help="Build Silver measurement table messungen_komplett.csv")
    sub.add_parser("gold-data-full", help="Build Gold hourly dataset data_full.csv")
    sub.add_parser("gold-masterdata", help="Aggregate data_full.csv to daily masterdata.csv")
    sub.add_parser("gold-daily-dataset", help="Build daily-grain dataset for modeling")

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

    sub.add_parser("run-all", help="Run Silver + Gold + (optional) training")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = get_paths(load_dotenv=True)

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

    if args.cmd == "train-ecoli":
        train_ecoli_predictability(
            paths,
            data_path=args.data,
            out_dir=args.out_dir,
            target=args.target,
            test_fraction=args.test_fraction,
            log_target=args.log_target,
            export_ecoli_only=args.export_ecoli_only,
            mlflow_enabled=args.mlflow,
            mlflow_experiment=args.mlflow_experiment,
            mlflow_run_name=args.mlflow_run_name,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_log_model=(not args.no_mlflow_model),
        )
        return 0

    if args.cmd == "run-all":
        build_silver_weather(paths)
        build_messungen_komplett(paths)
        build_data_full(paths)
        build_masterdata_daily(paths)
        build_daily_gold_dataset(paths)
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")
