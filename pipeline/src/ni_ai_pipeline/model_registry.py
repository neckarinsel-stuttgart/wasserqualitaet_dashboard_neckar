"""Model version registry: one CSV row per trained model, version counts up by 1.

File: <gold_datasets_dir>/model_versions.csv with columns
version, trained_at, training_data_until. The last row is the model currently in use.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ni_ai_pipeline.paths import PathConfig

REGISTRY_FILENAME = "model_versions.csv"
REGISTRY_COLUMNS = ["version", "trained_at", "training_data_until"]


def _registry_path(paths: PathConfig):
    return paths.gold_datasets_dir / REGISTRY_FILENAME


def _read_registry(paths: PathConfig) -> pd.DataFrame:
    path = _registry_path(paths)
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    df = pd.read_csv(path, dtype=str)
    df["version"] = pd.to_numeric(df["version"], errors="coerce")
    return df.dropna(subset=["version"])


def append_model_version(paths: PathConfig, *, trained_at: str, training_data_until: str) -> int:
    """Append a row for a newly trained model and return its version number."""

    df = _read_registry(paths)
    version = int(df["version"].max()) + 1 if not df.empty else 1
    row = pd.DataFrame(
        [{"version": version, "trained_at": trained_at, "training_data_until": training_data_until}]
    )
    out = pd.concat([df, row], ignore_index=True)[REGISTRY_COLUMNS]
    out["version"] = out["version"].astype(int)
    path = _registry_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return version


def read_latest_model_version(paths: PathConfig) -> dict[str, Any] | None:
    """Return the newest registry row, or None if there is no registry yet."""

    try:
        df = _read_registry(paths)
    except Exception:
        return None
    if df.empty:
        return None
    latest = df.sort_values("version").iloc[-1]

    def _text(value: Any) -> str | None:
        return None if pd.isna(value) else str(value)

    return {
        "version": int(latest["version"]),
        "trained_at": _text(latest.get("trained_at")),
        "training_data_until": _text(latest.get("training_data_until")),
    }
