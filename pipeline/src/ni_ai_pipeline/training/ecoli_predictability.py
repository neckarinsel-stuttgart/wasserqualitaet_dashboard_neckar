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

    y = _labels_to_binary(df[target_col])
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
    cv_accuracy: float
    cv_f1: float
    cv_macro_f1: float
    cv_balanced_accuracy: float
    cv_roc_auc: float
    holdout_accuracy: float
    holdout_f1: float
    holdout_macro_f1: float
    holdout_balanced_accuracy: float
    holdout_roc_auc: float


def _build_model_metadata(
    *,
    target: str,
    feature_columns: list[str],
    dropped_entro_columns: list[str],
    dropped_non_feature_columns: list[str],
    best_model_name: str,
    data_path: Path,
    training_row_count: int,
    training_max_date: str | None,
) -> dict[str, Any]:
    return {
        "target": target,
        "feature_columns": feature_columns,
        "dropped_entro_columns": dropped_entro_columns,
        "dropped_non_feature_columns": dropped_non_feature_columns,
        "best_model": best_model_name,
        "training_data_path": str(data_path),
        "training_row_count": int(training_row_count),
        "training_max_date": training_max_date,
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
    return np.asarray(model.predict(x))


def _labels_to_binary(y: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(y, errors="coerce")
    valid = numeric.dropna()
    if not valid.empty and set(valid.unique()).issubset({0.0, 1.0}):
        return numeric.astype("boolean")
    out = pd.Series(pd.NA, index=numeric.index, dtype="boolean")
    out.loc[numeric.notna()] = numeric.loc[numeric.notna()].gt(0).to_numpy()
    return out


def _positive_scores(model: Any, x: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None

    proba = model.predict_proba(x)
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None

    classes_arr = np.asarray(classes)
    if classes_arr.ndim != 1:
        return None

    positive_idx = None
    for idx, cls in enumerate(classes_arr):
        if bool(cls) is True or cls == 1 or cls == "1":
            positive_idx = idx
            break
    if positive_idx is None:
        positive_idx = len(classes_arr) - 1

    return np.asarray(proba[:, positive_idx], dtype=float)


def _dynamic_class_weight_from_messungen(paths: PathConfig, *, ecoli_threshold: float) -> dict[int, float]:
    labels_path = paths.silver_messungen_dir / "messungen_komplett.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Dynamic class weight requested, but missing labels file: {labels_path}"
        )

    df = pd.read_csv(labels_path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "ecoli" not in cols:
        raise KeyError(
            f"Dynamic class weight requested, but column 'ecoli' not found in {labels_path}. "
            f"Columns: {list(df.columns)}"
        )

    ecoli = pd.to_numeric(df[cols["ecoli"]], errors="coerce").dropna()
    if ecoli.empty:
        raise RuntimeError(
            f"Dynamic class weight requested, but no numeric ecoli values found in {labels_path}."
        )

    pos_count = int((ecoli <= float(ecoli_threshold)).sum())
    neg_count = int((ecoli > float(ecoli_threshold)).sum())
    n_total = pos_count + neg_count

    if pos_count == 0 or neg_count == 0:
        raise RuntimeError(
            "Dynamic class weight requested, but one class is empty at threshold "
            f"{ecoli_threshold}: pos={pos_count}, neg={neg_count}."
        )

    # sklearn-style balanced weighting: n_samples / (n_classes * n_samples_class)
    return {
        0: float(n_total / (2.0 * neg_count)),
        1: float(n_total / (2.0 * pos_count)),
    }


def train_ecoli_predictability(
    paths: PathConfig,
    *,
    data_path: Path | None = None,
    out_dir: Path | None = None,
    target: str = "pos_neg",
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
    class_weight_negative: float = 0.8,
    class_weight_positive: float = 0.2,
    auto_class_weight: bool = False,
    class_weight_threshold: float = 500.0,
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
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
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

    training_max_date: str | None = None
    if "date" in df.columns:
        training_dates = pd.to_datetime(df.loc[valid, "date"], errors="coerce")
        training_dates = training_dates.dropna()
        if not training_dates.empty:
            training_max_date = pd.Timestamp(training_dates.max()).strftime("%Y-%m-%d")

    y = y.astype(bool)

    if len(y) < 10:
        raise RuntimeError(f"Not enough label rows to evaluate (n={len(y)}).")

    if y.nunique() < 2:
        raise RuntimeError("Need at least two target classes to train a classifier.")

    n = len(y)
    holdout_n = max(1, int(math.ceil(n * test_fraction)))
    train_n = n - holdout_n
    if train_n < 5:
        raise RuntimeError(f"Holdout too large for dataset size (n={n}, holdout_n={holdout_n}).")

    x_train, x_test = x.iloc[:train_n], x.iloc[train_n:]
    y_train, y_test = y.iloc[:train_n], y.iloc[train_n:]

    if auto_class_weight:
        class_weight = _dynamic_class_weight_from_messungen(
            paths,
            ecoli_threshold=class_weight_threshold,
        )
        class_weight_negative = float(class_weight[0])
        class_weight_positive = float(class_weight[1])
    else:
        if class_weight_negative <= 0 or class_weight_positive <= 0:
            raise ValueError(
                "class_weight_negative and class_weight_positive must be > 0 "
                f"(got {class_weight_negative}, {class_weight_positive})."
            )

        class_weight = {0: float(class_weight_negative), 1: float(class_weight_positive)}

    pre = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])

    models: list[tuple[str, Any]] = [
        (
            "logistic_l2",
            LogisticRegression(
                C=1.0,
                class_weight=class_weight,
                solver="liblinear",
                max_iter=2000,
                random_state=0,
            ),
        ),
        (
            "ridge_classifier_weighted_a1",
            RidgeClassifier(
                alpha=1.0,
                class_weight=class_weight,
                random_state=0,
            ),
        ),
        (
            "ridge_classifier_weighted_a5",
            RidgeClassifier(
                alpha=5.0,
                class_weight=class_weight,
                random_state=0,
            ),
        ),
        (
            "random_forest_weighted",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=0,
                n_jobs=-1,
            ),
        ),
    ]

    n_splits = min(5, max(2, train_n // 8))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results: list[ModelResult] = []

    for name, reg in models:
        cv_accuracy_scores: list[float] = []
        cv_f1_scores: list[float] = []
        cv_macro_f1_scores: list[float] = []
        cv_balanced_accuracy_scores: list[float] = []
        cv_roc_auc_scores: list[float] = []

        for tr_idx, va_idx in tscv.split(x_train):
            xt, xv = x_train.iloc[tr_idx], x_train.iloc[va_idx]
            yt = y_train.iloc[tr_idx].astype(bool).to_numpy()
            yv = y_train.iloc[va_idx].astype(bool).to_numpy()

            fold_reg = reg if len(np.unique(yt)) > 1 else DummyClassifier(strategy="most_frequent")
            fold_pipe = Pipeline(steps=[("pre", pre), ("model", fold_reg)])

            fold_pipe.fit(xt, yt)
            pred = np.asarray(fold_pipe.predict(xv), dtype=bool)

            cv_accuracy_scores.append(accuracy_score(yv, pred))
            cv_f1_scores.append(f1_score(yv, pred, zero_division=0))
            cv_macro_f1_scores.append(f1_score(yv, pred, average="macro", zero_division=0))
            cv_balanced_accuracy_scores.append(balanced_accuracy_score(yv, pred))

            scores = _positive_scores(fold_pipe, xv)
            if scores is not None and len(np.unique(yv)) > 1:
                cv_roc_auc_scores.append(roc_auc_score(yv, scores))

        train_y_bool = y_train.astype(bool).to_numpy()
        final_reg = reg if len(np.unique(train_y_bool)) > 1 else DummyClassifier(strategy="most_frequent")
        final_pipe = Pipeline(steps=[("pre", pre), ("model", final_reg)])
        final_pipe.fit(x_train, train_y_bool)
        holdout_pred = np.asarray(final_pipe.predict(x_test), dtype=bool)
        holdout_scores = _positive_scores(final_pipe, x_test)

        results.append(
            ModelResult(
                name=name,
                cv_accuracy=float(np.mean(cv_accuracy_scores)),
                cv_f1=float(np.mean(cv_f1_scores)),
                cv_macro_f1=float(np.mean(cv_macro_f1_scores)),
                cv_balanced_accuracy=float(np.mean(cv_balanced_accuracy_scores)),
                cv_roc_auc=float(np.mean(cv_roc_auc_scores)) if cv_roc_auc_scores else float("nan"),
                holdout_accuracy=accuracy_score(y_test.astype(bool).to_numpy(), holdout_pred),
                holdout_f1=f1_score(y_test.astype(bool).to_numpy(), holdout_pred, zero_division=0),
                holdout_macro_f1=f1_score(
                    y_test.astype(bool).to_numpy(), holdout_pred, average="macro", zero_division=0
                ),
                holdout_balanced_accuracy=balanced_accuracy_score(
                    y_test.astype(bool).to_numpy(), holdout_pred
                ),
                holdout_roc_auc=(
                    roc_auc_score(y_test.astype(bool).to_numpy(), holdout_scores)
                    if holdout_scores is not None and len(np.unique(y_test.to_numpy())) > 1
                    else float("nan")
                ),
            )
        )

    best = max(
        results,
        key=lambda r: (
            r.holdout_macro_f1,
            r.holdout_balanced_accuracy,
            r.holdout_f1,
            r.holdout_accuracy,
        ),
    )

    best_reg = dict(models)[best.name]
    best_pipe = Pipeline(steps=[("pre", pre), ("model", best_reg)])
    train_y_bool = y_train.astype(bool).to_numpy()
    if len(np.unique(train_y_bool)) < 2:
        best_pipe = Pipeline(steps=[("pre", pre), ("model", DummyClassifier(strategy="most_frequent"))])
    best_pipe.fit(x_train, train_y_bool)

    try:
        perm = permutation_importance(
            best_pipe,
            x_test,
            y_test.astype(bool).to_numpy(),
            n_repeats=50,
            random_state=0,
            scoring="f1",
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

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        ["holdout_macro_f1", "holdout_balanced_accuracy", "holdout_f1", "holdout_accuracy"],
        ascending=[False, False, False, False],
    )
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
        f.write(f"Class weight mode: {'auto_from_messungen' if auto_class_weight else 'manual'}\n")
        if auto_class_weight:
            f.write(f"Class weight ecoli threshold: {class_weight_threshold:.3f}\n")
        f.write(
            "Class weights "
            f"(label=0, label=1): ({class_weight_negative:.3f}, {class_weight_positive:.3f})\n\n"
        )

        f.write("Target summary (train):\n")
        f.write(f"  positive_rate={float(np.mean(y_train.astype(bool))):.3f}\n\n")

        f.write("Model comparison (sorted by holdout macro-F1, balanced accuracy, F1, accuracy):\n")
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
            training_row_count=len(y),
            training_max_date=training_max_date,
        )
        model_metadata["task"] = "binary_classification"
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
                    "class_weight_mode": "auto_from_messungen" if auto_class_weight else "manual",
                    "class_weight_threshold": float(class_weight_threshold),
                    "class_weight_negative": float(class_weight_negative),
                    "class_weight_positive": float(class_weight_positive),
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
                        prefix + "cv_accuracy": r.cv_accuracy,
                        prefix + "cv_f1": r.cv_f1,
                        prefix + "cv_macro_f1": r.cv_macro_f1,
                        prefix + "cv_balanced_accuracy": r.cv_balanced_accuracy,
                        prefix + "cv_roc_auc": r.cv_roc_auc,
                        prefix + "holdout_accuracy": r.holdout_accuracy,
                        prefix + "holdout_f1": r.holdout_f1,
                        prefix + "holdout_macro_f1": r.holdout_macro_f1,
                        prefix + "holdout_balanced_accuracy": r.holdout_balanced_accuracy,
                        prefix + "holdout_roc_auc": r.holdout_roc_auc,
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
