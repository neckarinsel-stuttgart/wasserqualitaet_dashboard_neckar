from __future__ import annotations

import pandas as pd

from ni_ai_pipeline.api_payloads import build_last_30d_plot_payload


def test_build_plot_payload_with_days_none_returns_full_history() -> None:
    df = pd.DataFrame(
        [
            {"date": "2026-01-01", "x": 1.0},
            {"date": "2026-01-05", "x": 2.0},
            {"date": "2026-01-10", "x": 3.0},
        ]
    )

    payload = build_last_30d_plot_payload(
        df,
        plot_cols=["x"],
        site_id="default",
        date_col="date",
        days=None,
    )

    assert payload["timestamps"] == ["2026-01-01", "2026-01-05", "2026-01-10"]
    assert len(payload["rows"]) == 3
