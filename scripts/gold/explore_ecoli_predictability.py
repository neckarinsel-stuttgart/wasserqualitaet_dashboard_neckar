import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


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


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    if date_col not in df.columns:
        return df

    dt = pd.to_datetime(df[date_col], errors="coerce")
    # Normalize so time-of-day does not leak in
    dt = dt.dt.tz_localize(None)

    doy = dt.dt.dayofyear.astype(float)
    # Use periodic encoding; handles missing by leaving NaNs
    df["day_of_year"] = doy
    df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    df["year"] = dt.dt.year.astype(float)
    df["month"] = dt.dt.month.astype(float)
    return df


def drop_entro_everywhere(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    lower = {c: str(c).lower() for c in df.columns}
    entro_cols = [c for c, cl in lower.items() if "entro" in cl]
    return df.drop(columns=entro_cols, errors="ignore"), entro_cols


def select_features(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    y = pd.to_numeric(df[target_col], errors="coerce")

    X = df.drop(columns=[target_col, *drop_cols], errors="ignore")

    # Keep only numeric feature columns
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[numeric_cols]

    # Drop columns that are entirely missing (these will otherwise trigger imputer warnings)
    X = X.dropna(axis=1, how="all")

    # Final safety: ensure entro is not present
    if any("entro" in str(c).lower() for c in X.columns):
        raise RuntimeError("Entro is present in feature columns after filtering. Aborting.")

    return X, y


@dataclass
class ModelResult:
    name: str
    cv_mae: float
    cv_rmse: float
    cv_r2: float
    holdout_mae: float
    holdout_rmse: float
    holdout_r2: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore predictability of ecoli from daily Gold features (entro excluded).")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/gold/datasets/gold_daily_dataset.csv"),
        help="Path to gold_daily_dataset.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/gold/datasets"),
        help="Directory to write report/artifacts",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="ecoli",
        help="Target column name (default: ecoli)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Holdout fraction from the end of the time series (default: 0.2)",
    )
    parser.add_argument(
        "--log-target",
        action="store_true",
        help="Evaluate models on log1p(target) with inverse transform back to original scale for metrics.",
    )
    parser.add_argument(
        "--export-ecoli-only",
        action="store_true",
        help="Write a copy of the dataset with all 'entro' columns removed.",
    )

    args = parser.parse_args()

    # Local imports so the script can still be inspected without sklearn installed.
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline

    data_path = args.data
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df = add_time_features(df, date_col="date")

    df, dropped_entro_cols = drop_entro_everywhere(df)

    if args.export_ecoli_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ecoli_only_path = args.out_dir / "gold_daily_dataset_ecoli_only.csv"
        df.to_csv(ecoli_only_path, index=False)

    # Columns that are identifiers / not features
    base_drop = ["site_id", "date", "n_samples"]

    X, y = select_features(df, target_col=args.target, drop_cols=base_drop)

    # Sort by date if present
    if "date" in df.columns:
        order = pd.to_datetime(df["date"], errors="coerce")
        sort_idx = np.argsort(order.values)
        X = X.iloc[sort_idx]
        y = y.iloc[sort_idx]

    # Drop rows where target is missing
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    if len(y) < 10:
        raise RuntimeError(f"Not enough label rows to evaluate (n={len(y)}).")

    n = len(y)
    holdout_n = max(1, int(math.ceil(n * args.test_fraction)))
    train_n = n - holdout_n
    if train_n < 5:
        raise RuntimeError(f"Holdout too large for dataset size (n={n}, holdout_n={holdout_n}).")

    X_train, X_test = X.iloc[:train_n], X.iloc[train_n:]
    y_train, y_test = y.iloc[:train_n], y.iloc[train_n:]

    pre = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )

    def wrap_target(reg):
        if not args.log_target:
            return reg
        return TransformedTargetRegressor(
            regressor=reg,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    models: List[Tuple[str, Any]] = [
        ("dummy_mean", DummyRegressor(strategy="mean")),
        ("ridge", Ridge(alpha=1.0, random_state=0)),
        ("rf", RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)),
        ("hgb", HistGradientBoostingRegressor(random_state=0)),
    ]

    # Time-series CV on the training portion only
    n_splits = min(5, max(2, train_n // 8))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results: List[ModelResult] = []

    for name, reg in models:
        pipe = Pipeline(steps=[("pre", pre), ("model", wrap_target(reg))])

        cv_mae_scores: List[float] = []
        cv_rmse_scores: List[float] = []
        cv_r2_scores: List[float] = []

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train), start=1):
            Xt, Xv = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            yt, yv = y_train.iloc[tr_idx].to_numpy(), y_train.iloc[va_idx].to_numpy()

            pipe.fit(Xt, yt)
            pred = pipe.predict(Xv)

            cv_mae_scores.append(_mae(yv, pred))
            cv_rmse_scores.append(_rmse(yv, pred))
            cv_r2_scores.append(_r2(yv, pred))

        # Holdout evaluation (final fit)
        pipe.fit(X_train, y_train.to_numpy())
        holdout_pred = pipe.predict(X_test)

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

    # Pick best model by holdout MAE
    best = min(results, key=lambda r: r.holdout_mae)

    # Refit best model for permutation importance on holdout
    best_reg = dict(models)[best.name]
    best_pipe = Pipeline(steps=[("pre", pre), ("model", wrap_target(best_reg))])
    best_pipe.fit(X_train, y_train.to_numpy())

    try:
        perm = permutation_importance(
            best_pipe,
            X_test,
            y_test.to_numpy(),
            n_repeats=50,
            random_state=0,
            scoring="neg_mean_absolute_error",
        )
        importances = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
    except Exception as e:
        importances = pd.DataFrame({"error": [str(e)]})

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("holdout_mae")
    results_path = args.out_dir / "ecoli_predictability_results.csv"
    results_df.to_csv(results_path, index=False)

    importances_path = args.out_dir / "ecoli_feature_importance_permutation.csv"
    importances.to_csv(importances_path, index=False)

    report_path = args.out_dir / "ecoli_predictability_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ECOLI PREDICTABILITY REPORT\n")
        f.write("==========================\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Rows (with ecoli): {len(y)}\n")
        f.write(f"Features used: {X.shape[1]}\n")
        f.write(f"Dropped entro columns: {dropped_entro_cols}\n")
        f.write(f"Dropped non-feature cols: {base_drop + [args.target]}\n")
        f.write(f"Log target: {args.log_target}\n")
        if args.export_ecoli_only:
            f.write(f"Wrote ecoli-only dataset: {ecoli_only_path}\n")
        f.write(f"Train rows: {len(y_train)}\n")
        f.write(f"Holdout rows (last): {len(y_test)}\n")
        f.write(f"TimeSeriesSplit folds: {n_splits}\n\n")

        f.write("Target summary (train):\n")
        f.write(f"  mean={float(np.mean(y_train)):.3f}  std={float(np.std(y_train)):.3f}  min={float(np.min(y_train)):.3f}  max={float(np.max(y_train)):.3f}\n\n")

        f.write("Model comparison (sorted by holdout MAE):\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")

        f.write(f"Best model: {best.name}\n")
        f.write("\n")
        f.write(f"Wrote results CSV: {results_path}\n")
        f.write(f"Wrote importance CSV: {importances_path}\n")

    print(f"Wrote report: {report_path}")
    print(f"Wrote results: {results_path}")
    print(f"Wrote importances: {importances_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
