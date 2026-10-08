"""Domain-feature logistic model for P(ecoli > limit), prior-corrected to the recent regime.

2024 (different lab, much higher counts) is down-weighted; predicted probabilities are shifted
so the expected share of bad days matches ECOLI_TARGET_PRIOR instead of the training mix.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ni_ai_pipeline.paths import PathConfig
from ni_ai_pipeline.training.ecoli_predictability import TRAINING_START_DATE

DOMAIN_FEATURES = [
    "dom_rain_24h",
    "dom_rain_48h",
    "dom_rain_72h",
    "dom_rain_7d",
    "dom_rain_max1h_72h",
    "dom_sun_72h",
    "dom_airtemp_72h",
    "dom_doy_sin",
    "dom_doy_cos",
    "dom_turb_d1",
    "dom_cond_d1",
    "dom_o2_d1",
    "dom_wtemp_d1",
    "dom_turb_max3d",
    "dom_cond_drop7d",
]
MODEL_NAME = "domain_logit_prior_shift"
ECOLI_LIMIT = 500.0
TARGET_PRIOR = float(os.getenv("ECOLI_TARGET_PRIOR") or "0.20")
WARN_THRESHOLD = float(os.getenv("ECOLI_WARN_THRESHOLD") or "0.35")
WEIGHT_2024 = float(os.getenv("ECOLI_WEIGHT_2024") or "0.5")
LOGIT_C = 1.0


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


class PriorShiftedWarningModel:
    """predict() returns True when the day is expected to be <= ECOLI_LIMIT (pos_neg semantics)."""

    classes_ = np.array([False, True])

    def __init__(self, pipe, shift: float, threshold: float) -> None:
        self.pipe = pipe
        self.shift = shift
        self.threshold = threshold

    def proba_bad(self, x: pd.DataFrame) -> np.ndarray:
        z = self.pipe.decision_function(x) + self.shift
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = self.proba_bad(x)
        return np.column_stack([p, 1.0 - p])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.proba_bad(x) < self.threshold


def _fit(x: pd.DataFrame, y_bad: np.ndarray, weights: np.ndarray) -> PriorShiftedWarningModel:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=LOGIT_C, max_iter=5000),
    )
    pipe.fit(x, y_bad, logisticregression__sample_weight=weights)
    shift = _logit(TARGET_PRIOR) - _logit(float(np.average(y_bad, weights=weights)))
    return PriorShiftedWarningModel(pipe, shift, WARN_THRESHOLD)


def train_ecoli_domain(paths: PathConfig) -> PriorShiftedWarningModel:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    data_path = paths.gold_datasets_dir / "gold_daily_dataset.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df[pd.to_numeric(df["ecoli"], errors="coerce").notna()].sort_values("date").reset_index(drop=True)
    if TRAINING_START_DATE:
        df = df[df["date"] >= pd.Timestamp(TRAINING_START_DATE)].reset_index(drop=True)

    missing = [c for c in DOMAIN_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing domain features in {data_path}: {missing}. Rebuild gold first.")

    x = df[DOMAIN_FEATURES]
    y_bad = (pd.to_numeric(df["ecoli"]) > ECOLI_LIMIT).astype(int).to_numpy()
    years = df["date"].dt.year.to_numpy()
    weights = np.where(years == 2024, WEIGHT_2024, 1.0)
    use = weights > 0

    # Leave-one-year-out evaluation (reporting only; final model uses all rows)
    oof = np.full(len(df), np.nan)
    for yr in np.unique(years):
        te = years == yr
        tr = ~te & use
        if len(np.unique(y_bad[tr])) == 2:
            oof[te] = _fit(x[tr], y_bad[tr], weights[tr]).proba_bad(x[te])

    model = _fit(x[use], y_bad[use], weights[use])
    p_final = model.proba_bad(x)

    recent = (years >= 2025) & ~np.isnan(oof)
    eval_lines = []
    if recent.any() and len(np.unique(y_bad[recent])) == 2:
        yr_, pr_ = y_bad[recent], oof[recent]
        warn = pr_ >= WARN_THRESHOLD
        eval_lines = [
            f"LOYO on 2025+ (n={recent.sum()}, bad={yr_.sum()}): "
            f"AUC={roc_auc_score(yr_, pr_):.3f} Brier={brier_score_loss(yr_, pr_):.3f} "
            f"mean P(bad)={pr_.mean():.2f} actual={yr_.mean():.2f}",
            f"At threshold {WARN_THRESHOLD}: warnings={warn.sum()} "
            f"caught={int((warn & (yr_ == 1)).sum())}/{int(yr_.sum())} "
            f"false_alarms={int((warn & (yr_ == 0)).sum())}/{int((yr_ == 0).sum())}",
        ]

    out = paths.gold_datasets_dir
    with open(out / "ecoli_model.pkl", "wb") as f:
        pickle.dump(model, f)

    coef = pd.Series(model.pipe[-1].coef_[0], index=DOMAIN_FEATURES).sort_values(key=abs, ascending=False)
    metadata = {
        "task": "binary_classification",
        "target": "pos_neg",
        "best_model": MODEL_NAME,
        "ecoli_threshold": ECOLI_LIMIT,
        "warn_threshold_p_bad": WARN_THRESHOLD,
        "target_prior_bad": TARGET_PRIOR,
        "weight_2024": WEIGHT_2024,
        "training_start_date": TRAINING_START_DATE,
        "feature_columns": DOMAIN_FEATURES,
        "training_row_count": int(len(df)),
        "training_max_date": df["date"].max().strftime("%Y-%m-%d"),
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }
    (out / "ecoli_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pd.DataFrame(
        {"date": df["date"].dt.strftime("%Y-%m-%d"), "ecoli": df["ecoli"], "p_bad_loyo": oof.round(3),
         "p_bad_final": p_final.round(3)}
    ).to_csv(out / "ecoli_model_backtest.csv", index=False)

    report = [
        "ECOLI DOMAIN MODEL REPORT",
        "=========================",
        f"Model: {MODEL_NAME} (scaled logistic regression, C={LOGIT_C})",
        f"Rows: {len(df)} (bad >{ECOLI_LIMIT:g}: {int(y_bad.sum())}), weight 2024: {WEIGHT_2024}",
        f"Target prior P(bad): {TARGET_PRIOR}, warn threshold P(bad) >= {WARN_THRESHOLD}",
        *eval_lines,
        "",
        "Standardized coefficients (+ = more likely > limit):",
        coef.round(3).to_string(),
    ]
    (out / "ecoli_model_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return model
