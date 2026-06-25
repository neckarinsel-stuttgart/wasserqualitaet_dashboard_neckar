from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from ni_ai_pipeline.paths import PathConfig
from ni_ai_pipeline.training.daily_prediction import predict_latest_and_upsert


class _DummyModel:
    def predict(self, x: pd.DataFrame):
        return x[["f1", "f2"]].sum(axis=1).to_numpy()


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


def test_predict_latest_and_upsert(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.gold_datasets_dir.mkdir(parents=True, exist_ok=True)

    features_path = paths.gold_datasets_dir / "gold_daily_features.csv"
    pd.DataFrame(
        [
            {"site_id": "default", "date": "2026-06-18", "f1": 1.0, "f2": 2.0},
            {"site_id": "default", "date": "2026-06-19", "f1": 3.0, "f2": 4.0},
        ]
    ).to_csv(features_path, index=False)

    model_path = paths.gold_datasets_dir / "ecoli_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(_DummyModel(), f)

    metadata_path = paths.gold_datasets_dir / "ecoli_model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "target": "ecoli",
                "feature_columns": ["f1", "f2"],
                "best_model": "dummy",
            }
        ),
        encoding="utf-8",
    )

    predictions_path = predict_latest_and_upsert(
        paths,
        model_path=model_path,
        model_metadata_path=metadata_path,
        horizon_days=1,
    )

    out = pd.read_csv(predictions_path)
    assert len(out) == 1
    assert str(out.loc[0, "prediction_date"]) == "2026-06-20"
    assert bool(out.loc[0, "prediction"]) is True

    predict_latest_and_upsert(
        paths,
        model_path=model_path,
        model_metadata_path=metadata_path,
        horizon_days=1,
    )
    out2 = pd.read_csv(predictions_path)
    assert len(out2) == 1
