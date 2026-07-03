from __future__ import annotations

import json
from datetime import date
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


def build_last_30d_plot_payload(
    df: pd.DataFrame,
    *,
    plot_cols: Sequence[str],
    site_id: str | None,
    date_col: str = "date",
    end_date: str | pd.Timestamp | date | None = None,
    days: int | None = 30,
    strict_columns: bool = False,
) -> dict[str, Any]:
    """Build the same JSON payload shape as `scripts/gold/june_gold_feature_graphs.ipynb`,
    but for the most recent `days` rows by calendar day.

    Payload shape:
        {
          "year": int,
          "month": int,
          "site_id": str|None,
          "columns": [<plot col names>],
          "timestamps": ["YYYY-MM-DD", ...],
          "rows": [ {"date": "YYYY-MM-DD", <col>: <value|null>, ...}, ... ]
        }

    Notes:
    - Output key names match the notebook (`year`, `month`, `site_id`, `columns`, `timestamps`, `rows`).
    - NaN/inf values are converted to JSON null.
    - `year`/`month` are derived from the chosen `end_date` (usually the most recent date in the data).
    """

    if days is not None and days <= 0:
        raise ValueError(f"days must be positive when provided, got {days}")

    if date_col not in df.columns:
        raise KeyError(f"Missing date column '{date_col}'. Available: {list(df.columns)}")

    # Mirror notebook behavior: only include columns that exist (unless strict requested).
    missing_cols = [c for c in plot_cols if c not in df.columns]
    if missing_cols and strict_columns:
        raise KeyError(f"Missing plot columns: {missing_cols}")

    effective_plot_cols = [c for c in plot_cols if c in df.columns]

    dt = pd.to_datetime(df[date_col], errors="coerce")
    dt = dt.dt.tz_localize(None) if hasattr(dt.dt, "tz_localize") else dt

    if end_date is None:
        end_ts = pd.to_datetime(dt.max(), errors="coerce")
    else:
        end_ts = pd.to_datetime(end_date, errors="coerce")

    if pd.isna(end_ts):
        raise ValueError("Could not determine end_date (no valid dates found).")

    end_ts = pd.Timestamp(end_ts).tz_localize(None).normalize()
    if days is None:
        mask = dt <= end_ts
    else:
        start_ts = (end_ts - pd.Timedelta(days=days - 1)).normalize()
        mask = (dt >= start_ts) & (dt <= end_ts)
    last = df.loc[mask, [date_col, *effective_plot_cols]].copy()
    last[date_col] = pd.to_datetime(last[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    last = last.sort_values(date_col)

    # --- JSON output (same as notebook) ---
    json_df = last.rename(columns={date_col: "date"})
    json_df["date"] = pd.to_datetime(json_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    json_df = json_df.replace([np.inf, -np.inf], np.nan).astype(object)
    json_df = json_df.where(pd.notna(json_df), None)

    def _json_safe(value: Any) -> Any:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    rows = [
        {k: _json_safe(v) for k, v in row.items()}
        for row in json_df.to_dict(orient="records")
    ]

    payload: dict[str, Any] = {
        "year": int(end_ts.year),
        "month": int(end_ts.month),
        "site_id": site_id,
        "columns": list(effective_plot_cols),
        "timestamps": json_df["date"].tolist(),
        "rows": rows,
    }
    return payload


def dumps_plot_payload(payload: dict[str, Any], *, indent: int = 2) -> str:
    """Dump payload to JSON text using the same json.dumps settings as the notebook."""

    return json.dumps(payload, ensure_ascii=False, indent=indent, allow_nan=False)


def build_last_30d_plot_payload_json(
    df: pd.DataFrame,
    *,
    plot_cols: Sequence[str],
    site_id: str | None,
    date_col: str = "date",
    end_date: str | pd.Timestamp | date | None = None,
    days: int | None = 30,
    strict_columns: bool = False,
    indent: int = 2,
) -> str:
    """Convenience wrapper: build payload then return JSON text."""

    payload = build_last_30d_plot_payload(
        df,
        plot_cols=plot_cols,
        site_id=site_id,
        date_col=date_col,
        end_date=end_date,
        days=days,
        strict_columns=strict_columns,
    )
    return dumps_plot_payload(payload, indent=indent)
