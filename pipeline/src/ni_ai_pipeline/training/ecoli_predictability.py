from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import pickle

import numpy as np
import pandas as pd

from ni_ai_pipeline.paths import PathConfig


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def add_time_features(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if date_col not in out.columns:
        return out

    dt = pd.to_datetime(out[date_col], errors="coerce")
    dt = dt.dt.tz_localize(None)

    doy = dt.dt.dayofyear.astype(float)
    out["day_of_year"] = doy
    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    out["year"] = dt.dt.year.astype(float)
    out["month"] = dt.dt.month.astype(float)
    return out


def drop_entro_everywhere(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    lower = {c: str(c).lower() for c in df.columns}
    entro_cols = [c for c, cl in lower.items() if "entro" in cl]
    return df.drop(columns=entro_cols, errors="ignore"), entro_cols


def select_features(
    df: pd.DataFrame,
    *,
    target_col: str,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    y = pd.to_numeric(df[target_col], errors="coerce")
    x = df.drop(columns=[target_col, *drop_cols], errors="ignore")

    numeric_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    x = x[numeric_cols]
    x = x.dropna(axis=1, how="all")

    if any("entro" in str(c).lower() for c in x.columns):
        raise RuntimeError("Entro is present in feature columns after filtering. Aborting.")

    return x, y


@dataclass
class ModelResult:
    name: str
    cv_mae: float
    cv_rmse: float
    cv_r2: float
    holdout_mae: float
    holdout_rmse: float
    holdout_r2: float


def _build_model_metadata(
    *,
    target: str,
    feature_columns: list[str],
    dropped_entro_columns: list[str],
    dropped_non_feature_columns: list[str],
    best_model_name: str,
    data_path: Path,
) -> dict[str, Any]:
    return {
        "target": target,
        "feature_columns": feature_columns,
        "dropped_entro_columns": dropped_entro_columns,
        "dropped_non_feature_columns": dropped_non_feature_columns,
        "best_model": best_model_name,
        "training_data_path": str(data_path),
    }


def load_saved_model(model_path: Path) -> Any:
    """Load a previously saved sklearn-compatible model pipeline."""

    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_model_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load JSON metadata stored with a trained model artifact."""

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def predict_with_saved_model(
    model: Any,
    features: pd.DataFrame,
    *,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Predict with a saved model while enforcing saved feature order/schema.

    Missing features are added as NaN so the model's imputer can handle them.
    """

    expected = [str(c) for c in metadata.get("feature_columns", [])]
    if not expected:
        raise ValueError("Model metadata does not contain feature_columns.")

    x = features.copy()
    for c in expected:
        if c not in x.columns:
            x[c] = np.nan

    x = x[expected]
    return np.asarray(model.predict(x), dtype=float)


def train_ecoli_predictability(
    paths: PathConfig,
    *,
    data_path: Path | None = None,
    out_dir: Path | None = None,
    target: str = "ecoli",
    test_fraction: float = 0.2,
    log_target: bool = False,
    export_ecoli_only: bool = False,
    save_model: bool = True,
    model_path: Path | None = None,
    model_metadata_path: Path | None = None,
    mlflow_enabled: bool = False,
    mlflow_experiment: str = "ni-ai",
    mlflow_run_name: str | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_log_model: bool = True,
) -> None:
    """Evaluate simple models for predicting ecoli from daily Gold features.

    Writes:
    - `ecoli_predictability_results.csv`
    - `ecoli_feature_importance_permutation.csv`
    - `ecoli_predictability_report.txt`

    Requires optional dependency: scikit-learn (`pipeline[train]`).
    """

    # Local imports so the pipeline can be used without sklearn.
    try:
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import Pipeline
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for training. Install with: pip install -e ./pipeline[train]"
        ) from exc

    if data_path is None:
        data_path = paths.gold_datasets_dir / "gold_daily_dataset.csv"
    if out_dir is None:
        out_dir = paths.gold_datasets_dir

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df = add_time_features(df, date_col="date")

    df, dropped_entro_cols = drop_entro_everywhere(df)

    if export_ecoli_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        ecoli_only_path = out_dir / "gold_daily_dataset_ecoli_only.csv"
        df.to_csv(ecoli_only_path, index=False)

    base_drop = ["site_id", "date", "n_samples"]

    x, y = select_features(df, target_col=target, drop_cols=base_drop)

    if "date" in df.columns:
        order = pd.to_datetime(df["date"], errors="coerce")
        sort_idx = np.argsort(order.values)
        x = x.iloc[sort_idx]
        y = y.iloc[sort_idx]

    valid = y.notna()
    x = x.loc[valid]
    y = y.loc[valid]

    if len(y) < 10:
        raise RuntimeError(f"Not enough label rows to evaluate (n={len(y)}).")

    n = len(y)
    holdout_n = max(1, int(math.ceil(n * test_fraction)))
    train_n = n - holdout_n
    if train_n < 5:
        raise RuntimeError(f"Holdout too large for dataset size (n={n}, holdout_n={holdout_n}).")

    x_train, x_test = x.iloc[:train_n], x.iloc[train_n:]
    y_train, y_test = y.iloc[:train_n], y.iloc[train_n:]

    pre = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])

    def wrap_target(reg: Any) -> Any:
        if not log_target:
            return reg
        return TransformedTargetRegressor(
            regressor=reg,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    models: list[tuple[str, Any]] = [
        ("dummy_mean", DummyRegressor(strategy="mean")),
        ("ridge", Ridge(alpha=1.0, random_state=0)),
        ("rf", RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)),
        ("hgb", HistGradientBoostingRegressor(random_state=0)),
    ]

    n_splits = min(5, max(2, train_n // 8))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results: list[ModelResult] = []

    for name, reg in models:
        pipe = Pipeline(steps=[("pre", pre), ("model", wrap_target(reg))])

        cv_mae_scores: list[float] = []
        cv_rmse_scores: list[float] = []
        cv_r2_scores: list[float] = []

        for tr_idx, va_idx in tscv.split(x_train):
            xt, xv = x_train.iloc[tr_idx], x_train.iloc[va_idx]
            yt = y_train.iloc[tr_idx].to_numpy()
            yv = y_train.iloc[va_idx].to_numpy()

            pipe.fit(xt, yt)
            pred = pipe.predict(xv)

            cv_mae_scores.append(_mae(yv, pred))
            cv_rmse_scores.append(_rmse(yv, pred))
            cv_r2_scores.append(_r2(yv, pred))

        pipe.fit(x_train, y_train.to_numpy())
        holdout_pred = pipe.predict(x_test)

        results.append(
            ModelResult(
                name=name,
                cv_mae=float(np.mean(cv_mae_scores)),
                cv_rmse=float(np.mean(cv_rmse_scores)),
                cv_r2=float(np.mean(cv_r2_scores)),
                holdout_mae=_mae(y_test.to_numpy(), holdout_pred),
                holdout_rmse=_rmse(y_test.to_numpy(), holdout_pred),
                holdout_r2=_r2(y_test.to_numpy(), holdout_pred),
            )
        )

    best = min(results, key=lambda r: r.holdout_mae)

    best_reg = dict(models)[best.name]
    best_pipe = Pipeline(steps=[("pre", pre), ("model", wrap_target(best_reg))])
    best_pipe.fit(x_train, y_train.to_numpy())

    try:
        perm = permutation_importance(
            best_pipe,
            x_test,
            y_test.to_numpy(),
            n_repeats=50,
            random_state=0,
            scoring="neg_mean_absolute_error",
        )
        importances = pd.DataFrame(
            {
                "feature": x.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
    except Exception as exc:  # pragma: no cover
        importances = pd.DataFrame({"error": [str(exc)]})

    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("holdout_mae")
    results_path = out_dir / "ecoli_predictability_results.csv"
    results_df.to_csv(results_path, index=False)

    importances_path = out_dir / "ecoli_feature_importance_permutation.csv"
    importances.to_csv(importances_path, index=False)

    report_path = out_dir / "ecoli_predictability_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ECOLI PREDICTABILITY REPORT\n")
        f.write("==========================\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Rows (with {target}): {len(y)}\n")
        f.write(f"Features used: {x.shape[1]}\n")
        f.write(f"Dropped entro columns: {dropped_entro_cols}\n")
        f.write(f"Dropped non-feature cols: {base_drop + [target]}\n")
        f.write(f"Log target: {log_target}\n")
        if export_ecoli_only:
            f.write(f"Wrote ecoli-only dataset: {out_dir / 'gold_daily_dataset_ecoli_only.csv'}\n")
        f.write(f"Train rows: {len(y_train)}\n")
        f.write(f"Holdout rows (last): {len(y_test)}\n")
        f.write(f"TimeSeriesSplit folds: {n_splits}\n\n")

        f.write("Target summary (train):\n")
        f.write(
            "  mean={:.3f}  std={:.3f}  min={:.3f}  max={:.3f}\n\n".format(
                float(np.mean(y_train)),
                float(np.std(y_train)),
                float(np.min(y_train)),
                float(np.max(y_train)),
            )
        )

        f.write("Model comparison (sorted by holdout MAE):\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")

        f.write(f"Best model: {best.name}\n\n")
        f.write(f"Wrote results CSV: {results_path}\n")
        f.write(f"Wrote importance CSV: {importances_path}\n")

    print(f"Wrote: {report_path}")
    print(f"Wrote: {results_path}")
    print(f"Wrote: {importances_path}")

    if save_model:
        model_path = model_path or (out_dir / "ecoli_model.pkl")
        model_metadata_path = model_metadata_path or (out_dir / "ecoli_model_metadata.json")

        with open(model_path, "wb") as f:
            pickle.dump(best_pipe, f)

        model_metadata = _build_model_metadata(
            target=target,
            feature_columns=list(map(str, x.columns)),
            dropped_entro_columns=dropped_entro_cols,
            dropped_non_feature_columns=base_drop + [target],
            best_model_name=best.name,
            data_path=data_path,
        )
        model_metadata_path.write_text(
            json.dumps(model_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Wrote: {model_path}")
        print(f"Wrote: {model_metadata_path}")

    if mlflow_enabled:
        try:
            import importlib

            mlflow = importlib.import_module("mlflow")
            mlflow_sklearn = importlib.import_module("mlflow.sklearn")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "MLflow requested but not installed. Install with: pip install -e ./pipeline[mlflow]"
            ) from exc

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

        mlflow.set_experiment(mlflow_experiment)

        run_name = mlflow_run_name or f"train-ecoli-{best.name}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "data_path": str(data_path),
                    "out_dir": str(out_dir),
                    "target": target,
                    "test_fraction": test_fraction,
                    "log_target": log_target,
                    "export_ecoli_only": export_ecoli_only,
                    "n_rows": int(len(y)),
                    "n_features": int(x.shape[1]),
                    "tscv_splits": int(n_splits),
                    "best_model": best.name,
                }
            )

            for r in results:
                prefix = f"model_{r.name}_"
                mlflow.log_metrics(
                    {
                        prefix + "cv_mae": r.cv_mae,
                        prefix + "cv_rmse": r.cv_rmse,
                        prefix + "cv_r2": r.cv_r2,
                        prefix + "holdout_mae": r.holdout_mae,
                        prefix + "holdout_rmse": r.holdout_rmse,
                        prefix + "holdout_r2": r.holdout_r2,
                    }
                )

            # Log files we wrote to disk
            mlflow.log_artifact(str(results_path))
            mlflow.log_artifact(str(importances_path))
            mlflow.log_artifact(str(report_path))

            if export_ecoli_only:
                ecoli_only_path = out_dir / "gold_daily_dataset_ecoli_only.csv"
                if ecoli_only_path.exists():
                    mlflow.log_artifact(str(ecoli_only_path))

            cols_path = out_dir / "ecoli_feature_columns.json"
            cols_payload = {
                "feature_columns": list(map(str, x.columns)),
                "dropped_entro_columns": dropped_entro_cols,
                "dropped_non_feature_columns": base_drop + [target],
            }
            cols_path.write_text(json.dumps(cols_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(cols_path))

            if mlflow_log_model:
                mlflow_sklearn.log_model(best_pipe, artifact_path="model")
