from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ni_ai_pipeline import cli
from ni_ai_pipeline.paths import PathConfig
from ni_ai_pipeline.steps.silver_messungen import build_messungen_komplett


def _paths(tmp_path: Path) -> PathConfig:
    data_bronze = tmp_path / "data" / "bronze"
    data_silver = tmp_path / "data" / "silver"
    data_gold = tmp_path / "data" / "gold"
    return PathConfig(
        root=tmp_path,
        data_bronze=data_bronze,
        data_silver=data_silver,
        data_gold=data_gold,
        bronze_dwd_dir=data_bronze / "dwd",
        bronze_lubw_dir=data_bronze / "lubw",
        bronze_messungen_dir=data_bronze / "messungen",
        silver_weather_dir=data_silver / "weather",
        silver_messungen_dir=data_silver / "messungen",
        gold_datasets_dir=data_gold / "datasets",
        site_id="default",
    )


def test_build_messungen_komplett_reads_all_yearly_inputs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.bronze_messungen_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"DATUM": "01.01.2024", "ecoli": "100", "entro": "50"}]).to_csv(
        paths.bronze_messungen_dir / "messungen_2024.csv", index=False
    )
    pd.DataFrame([{"DATUM": "01.01.2025", "ecoli": "200", "entro": "60"}]).to_csv(
        paths.bronze_messungen_dir / "messungen_2025.csv", index=False
    )
    pd.DataFrame([{"DATUM": "01.01.2026", "ecoli": "300", "entro": "70"}]).to_csv(
        paths.bronze_messungen_dir / "messungen_2026.csv", index=False
    )

    out = build_messungen_komplett(paths)

    assert len(out) == 3
    assert str(out["datum"].max().date()) == "2026-01-01"
    assert (paths.silver_messungen_dir / "messungen_komplett.csv").exists()


def test_ensure_classifier_artifacts_retrains_when_data_is_newer(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    paths.gold_datasets_dir.mkdir(parents=True, exist_ok=True)

    # Freshly built dataset now includes 2026 labels.
    pd.DataFrame(
        [
            {"date": "2026-01-01", "pos_neg": 1},
            {"date": "2026-01-02", "pos_neg": 0},
            {"date": "2026-01-03", "pos_neg": 1},
        ]
    ).to_csv(paths.gold_datasets_dir / "gold_daily_dataset.csv", index=False)

    model_path = paths.gold_datasets_dir / "ecoli_model.pkl"
    model_path.write_bytes(b"placeholder")

    metadata_path = paths.gold_datasets_dir / "ecoli_model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "task": "binary_classification",
                "training_row_count": 2,
                "training_max_date": "2026-01-02",
            }
        ),
        encoding="utf-8",
    )

    called = {"value": False}

    def _fake_train_ecoli_predictability(_paths: PathConfig) -> None:
        called["value"] = True

    monkeypatch.setattr(cli, "train_ecoli_predictability", _fake_train_ecoli_predictability)

    cli._ensure_classifier_artifacts(
        paths,
        model_path=model_path,
        model_metadata_path=metadata_path,
    )

    assert called["value"] is True
