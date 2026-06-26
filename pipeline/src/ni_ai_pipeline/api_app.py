from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from ni_ai_pipeline.api_payloads import build_last_30d_plot_payload_json
from ni_ai_pipeline.paths import PathConfig, get_paths


TEMP_REQUESTED: list[str] = [
    "V_TE005_mean",
    "V_TE050_mean",
    "Ho_Ne_Temperatur",
    "TT_TU_mean",
]

WQ_REQUESTED_0_10: list[str] = [
    "We_Ne_ElektrischeLeitfaehigkeit",
    "Sauerstoff",
    "Ho_Ne_Truebung,quantitativ",
]

WQ_REQUESTED_RAW_SAME_AXIS: list[str] = [
    "pH_wert",
    "ABSF_STD_mean",
]

REQUESTED: list[str] = [
    *TEMP_REQUESTED,
    *WQ_REQUESTED_0_10,
    *WQ_REQUESTED_RAW_SAME_AXIS,
]


ALIASES: dict[str, str] = {
    # common typos / casing
    "FD_LBERG_MEAN": "FD_LBERG_mean",  
    "tim_since_last_rain_1mm": "time_since_last_rain_1mm",
    "air_temp_CCD_7d": "air_temp_CDD_7d",
    # temperature naming variants
    "V_TE_005mean": "V_TE005_mean",
    "V_TE_005_mean": "V_TE005_mean",
    "V_TE_050_mean": "V_TE050_mean",
    "Ho_Ne_Temperatur.": "Ho_Ne_Temperatur",
    # wq naming variants
    "ElektrischeLeitfaehigkeit": "We_Ne_ElektrischeLeitfaehigkeit",
    "elektrische leitfähigkeit": "We_Ne_ElektrischeLeitfaehigkeit",
    "Truebung": "Ho_Ne_Truebung,quantitativ",
    "Truebung,quantitativ": "Ho_Ne_Truebung,quantitativ",
    "Trübung": "Ho_Ne_Truebung,quantitativ",
    "PH_Wert": "pH_wert",
    "pH_Wert": "pH_wert",
    "Sauerstoffgehalt": "Sauerstoff",
}


def _flatten_requested(items: Any) -> list[str]:
    flat: list[str] = []
    for item in items:
        if isinstance(item, (list, tuple, set)):
            flat.extend([str(x) for x in item])
        else:
            flat.append(str(item))
    return flat


def _time_since_threshold(values: pd.Series, threshold: float) -> pd.Series:
    """Notebook-compatible helper (days since last value >= threshold)."""

    x = pd.to_numeric(values, errors="coerce")
    out = np.full(len(x), np.nan, dtype=float)
    counter: float | None = None
    for i, v in enumerate(x.to_numpy()):
        if np.isnan(v):
            out[i] = np.nan
            continue
        if v >= threshold:
            counter = 0.0
        else:
            counter = 1.0 if counter is None else counter + 1.0
        out[i] = counter
    return pd.Series(out, index=values.index, name=f"time_since_ge_{threshold}")


def _add_dry_spell_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Replicates `add_dry_spell_features` from the June notebook."""

    df = df.copy()
    if date_col not in df.columns:
        raise KeyError(f"Missing {date_col=} in df")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    df = df.sort_values(date_col)

    rain_col = None
    for c in ["R1_sum", "R1_mean", "R1_max"]:
        if c in df.columns:
            rain_col = c
            break
    if rain_col is None:
        return df

    rain = pd.to_numeric(df[rain_col], errors="coerce")

    df["wet_days_last_1d"] = (rain >= 0.1).astype(float)

    ts = _time_since_threshold(rain, threshold=1.0)
    df["time_since_last_rain_1mm"] = ts
    df["dry_days_since_rain_1mm"] = ts

    temp_col = "TT_TU_mean" if "TT_TU_mean" in df.columns else None
    if temp_col is not None:
        is_rain_event = (rain >= 1.0).fillna(False)
        dry_spell_id = is_rain_event.cumsum()
        df["mean_temp_dry_spell"] = (
            df[temp_col]
            .groupby(dry_spell_id)
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        df["mean_temp_dry_spell"] = np.nan

    if "air_temp_CDD_7d" not in df.columns and temp_col is not None:
        base = 18.0
        cdd = (pd.to_numeric(df[temp_col], errors="coerce") - base).clip(lower=0.0)
        df["air_temp_CDD_7d"] = cdd.rolling(7, min_periods=1).sum()

    return df


def _load_gold_plot_dataframe(paths: PathConfig) -> pd.DataFrame:
    """Load + merge weather+masterdata like `scripts/gold/june_gold_feature_graphs.ipynb`."""

    gold_daily_path = paths.gold_datasets_dir / "gold_daily_features.csv"
    if not gold_daily_path.exists():
        raise FileNotFoundError(f"Weather dataset not found: {gold_daily_path}")

    weather = pd.read_csv(gold_daily_path)

    if paths.site_id is not None and "site_id" in weather.columns:
        weather = weather.loc[weather["site_id"].astype(str) == str(paths.site_id)].copy()

    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.tz_localize(None)
    weather = weather.sort_values("date")

    weather = _add_dry_spell_features(weather, date_col="date")

    wq: pd.DataFrame | None = None
    masterdata_path = paths.gold_datasets_dir / "masterdata.csv"
    if masterdata_path.exists():
        md = pd.read_csv(masterdata_path)
        if "zeit" in md.columns:
            md["date"] = pd.to_datetime(md["zeit"], errors="coerce").dt.tz_localize(None)
            md["date"] = md["date"].dt.normalize()

            available = set(md.columns)
            md_cols: list[str] = []
            for c in [
                "ecoli_mean",
                "entro_mean",
                "We_Ne_ElektrischeLeitfaehigkeit_mean",
                "Ho_Ne_Truebung,quantitativ_mean",
                "Ho_Ne_Temperatur_mean",
                "pH_wert_mean",
                "pH_Wert_mean",
                "Ho_Ne_pH-Wert_mean",
                "Ho_Ne_pH_Wert_mean",
                "Sauerstoff_mean",
                "Ho_Ne_Sauerstoff_mean",
                "Ho_Ne_Sauerstoffgehalt,gelöst_mean",
                "Ho_Ne_Sauerstoffgehalt_geloest_mean",
            ]:
                if c in available:
                    md_cols.append(c)

            if md_cols:
                wq = md[["date", *md_cols]].copy()
                rename_map = {
                    "ecoli_mean": "ecoli",
                    "entro_mean": "entro",
                    "We_Ne_ElektrischeLeitfaehigkeit_mean": "We_Ne_ElektrischeLeitfaehigkeit",
                    "Ho_Ne_Truebung,quantitativ_mean": "Ho_Ne_Truebung,quantitativ",
                    "Ho_Ne_Temperatur_mean": "Ho_Ne_Temperatur",
                    "pH_wert_mean": "pH_wert",
                    "pH_Wert_mean": "pH_wert",
                    "Ho_Ne_pH-Wert_mean": "pH_wert",
                    "Ho_Ne_pH_Wert_mean": "pH_wert",
                    "Sauerstoff_mean": "Sauerstoff",
                    "Ho_Ne_Sauerstoff_mean": "Sauerstoff",
                    "Ho_Ne_Sauerstoffgehalt,gelöst_mean": "Sauerstoff",
                    "Ho_Ne_Sauerstoffgehalt_geloest_mean": "Sauerstoff",
                }
                wq = wq.rename(columns=rename_map)

    weather["date"] = weather["date"].dt.normalize()

    df = weather.copy()
    if wq is not None:
        df = df.merge(wq, on="date", how="left")

    return df


def _prediction_value_to_bool(
    value: Any,
    *,
    threshold: float = 0.5,
    target: str | None = None,
    ecoli_threshold: float = 1000.0,
) -> bool:
    """Convert prediction table values to boolean in a tolerant way."""

    target_token = str(target or "").strip().lower()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "t", "yes", "y", "1"}:
            return True
        if token in {"false", "f", "no", "n", "0"}:
            return False

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Could not convert prediction value to bool: {value!r}")
    if target_token == "ecoli":
        return bool(float(numeric) <= float(ecoli_threshold))
    return bool(float(numeric) >= float(threshold))


def _normalize_prediction_target_label(target: Any) -> str:
    token = str(target or "").strip().lower()
    if token == "pos_neg":
        return "ecoli"
    return token


def _json_compatible_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _rows_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({str(k): _json_compatible_value(v) for k, v in row.items()})
    return records


def create_app() -> FastAPI:
    # Expose API endpoints only via explicit routes (no auto docs/openapi routes).
    app = FastAPI(
        title="NI_AI API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/plot-data")
    def get_plot_data_last_30d() -> Response:
        """Return last-30-days plot payload JSON (same format as notebook export).

        Returned as an application/json download (no auth).
        """

        paths = get_paths(load_dotenv=True)
        try:
            df = _load_gold_plot_dataframe(paths)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        requested_flat = _flatten_requested(REQUESTED)
        resolved = [ALIASES.get(name, name) for name in requested_flat]

        plot_cols = [c for c in resolved if c in df.columns]

        # Cap turbidity at 25 like the notebook (prevents spikes dominating).
        turb_col = "Ho_Ne_Truebung,quantitativ"
        if turb_col in df.columns:
            df[turb_col] = pd.to_numeric(df[turb_col], errors="coerce").clip(upper=25.0)

        json_text = build_last_30d_plot_payload_json(
            df,
            plot_cols=plot_cols,
            site_id=paths.site_id,
            date_col="date",
            days=30,
            strict_columns=False,
            indent=2,
        )

        headers = {
            "Content-Disposition": 'attachment; filename="last_30d_plot_data.json"'
        }
        return Response(content=json_text, media_type="application/json", headers=headers)

    @app.get("/masterdata")
    def get_masterdata_csv() -> FileResponse:
        """Return the gold `masterdata.csv` as a file download (no auth)."""

        paths = get_paths(load_dotenv=True)
        masterdata_path: Path = paths.gold_datasets_dir / "masterdata.csv"
        if not masterdata_path.exists():
            raise HTTPException(status_code=404, detail=f"Not found: {masterdata_path}")

        return FileResponse(
            path=str(masterdata_path),
            media_type="text/csv",
            filename="masterdata.csv",
        )

    @app.get("/get_daily_prediction")
    def get_daily_prediction() -> dict[str, Any]:
        """Return latest prediction as boolean + current-day freshness flag."""

        paths = get_paths(load_dotenv=True)
        predictions_path = paths.gold_datasets_dir / "predictions.csv"
        if not predictions_path.exists():
            raise HTTPException(status_code=404, detail=f"Not found: {predictions_path}")

        try:
            df = pd.read_csv(predictions_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No rows in: {predictions_path}")

        if paths.site_id is not None and "site_id" in df.columns:
            df = df.loc[df["site_id"].astype(str) == str(paths.site_id)].copy()
            if df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No prediction rows for site_id={paths.site_id}",
                )

        if "prediction_date" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="predictions.csv is missing required column: prediction_date",
            )
        if "prediction" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="predictions.csv is missing required column: prediction",
            )

        df["prediction_date"] = pd.to_datetime(
            df["prediction_date"], errors="coerce"
        ).dt.tz_localize(None)
        df = df.dropna(subset=["prediction_date"])
        if df.empty:
            raise HTTPException(status_code=404, detail="No valid prediction_date rows found")

        sort_cols = ["prediction_date"]
        if "created_at_utc" in df.columns:
            df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], errors="coerce", utc=True)
            sort_cols.append("created_at_utc")

        latest = df.sort_values(sort_cols).iloc[-1]

        try:
            prediction_bool = _prediction_value_to_bool(
                latest["prediction"],
                target=(str(latest["target"]) if "target" in latest.index else None),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        latest_date = pd.Timestamp(latest["prediction_date"]).normalize()
        today = pd.Timestamp.now().normalize()
        is_current_day = bool(latest_date == today)

        payload: dict[str, Any] = {
            "prediction": prediction_bool,
            "is_current_day": is_current_day,
        }

        if "prediction_date" in latest.index:
            payload["prediction_date"] = pd.Timestamp(latest["prediction_date"]).strftime("%Y-%m-%d")
        if "site_id" in latest.index:
            payload["site_id"] = str(latest["site_id"])
        if "target" in latest.index:
            payload["target"] = _normalize_prediction_target_label(latest["target"])

        return payload

    @app.get("/get_current_weather")
    def get_current_weather() -> dict[str, Any]:
        """Return Stuttgart weather row for the current local hour."""

        paths = get_paths(load_dotenv=True)
        weather_path = paths.gold_datasets_dir / "stuttgart_weather.csv"
        if not weather_path.exists():
            raise HTTPException(status_code=404, detail=f"Not found: {weather_path}")

        try:
            df = pd.read_csv(weather_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No rows in: {weather_path}")

        if "weather_time_local" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="stuttgart_weather.csv is missing required column: weather_time_local",
            )

        if paths.site_id is not None and "site_id" in df.columns:
            df = df.loc[df["site_id"].astype(str) == str(paths.site_id)].copy()
            if df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No weather rows for site_id={paths.site_id}",
                )

        timezone = "Europe/Berlin"
        if "timezone" in df.columns:
            tz_values = df["timezone"].dropna().astype(str)
            if not tz_values.empty:
                timezone = tz_values.iloc[-1]

        now_local_hour = pd.Timestamp.now(tz=timezone).tz_localize(None).replace(
            minute=0, second=0, microsecond=0
        )

        df["weather_time_local"] = pd.to_datetime(df["weather_time_local"], errors="coerce")
        df = df.dropna(subset=["weather_time_local"])
        if df.empty:
            raise HTTPException(status_code=404, detail="No valid weather_time_local rows found")

        current_rows = df.loc[df["weather_time_local"] == now_local_hour].copy()
        if current_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No weather row for current hour: {now_local_hour.strftime('%Y-%m-%d %H:00:00')}",
            )

        sort_cols = ["weather_time_local"]
        if "created_at_utc" in current_rows.columns:
            current_rows["created_at_utc"] = pd.to_datetime(
                current_rows["created_at_utc"], errors="coerce", utc=True
            )
            sort_cols.append("created_at_utc")

        latest = current_rows.sort_values(sort_cols).iloc[-1]

        payload = {
            str(k): _json_compatible_value(v)
            for k, v in latest.to_dict().items()
        }
        payload["weather_time_local"] = now_local_hour.strftime("%Y-%m-%d %H:%M:%S")

        return payload

    @app.get("/get_last_30d_weather")
    @app.get("/get-last-30d-weather")
    def get_last_30d_weather() -> dict[str, Any]:
        """Return all available weather rows from stuttgart_weather.csv."""

        paths = get_paths(load_dotenv=True)
        weather_path = paths.gold_datasets_dir / "stuttgart_weather.csv"
        if not weather_path.exists():
            raise HTTPException(status_code=404, detail=f"Not found: {weather_path}")

        try:
            df = pd.read_csv(weather_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No rows in: {weather_path}")

        if "weather_time_local" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="stuttgart_weather.csv is missing required column: weather_time_local",
            )

        df["weather_time_local"] = pd.to_datetime(df["weather_time_local"], errors="coerce")
        df = df.dropna(subset=["weather_time_local"])
        if df.empty:
            raise HTTPException(status_code=404, detail="No valid weather_time_local rows found")

        window = df.copy()

        sort_cols = ["weather_time_local"]
        if "created_at_utc" in window.columns:
            window["created_at_utc"] = pd.to_datetime(
                window["created_at_utc"], errors="coerce", utc=True
            )
            sort_cols.append("created_at_utc")

        window = window.sort_values(sort_cols)
        min_weather_ts = pd.to_datetime(window["weather_time_local"], errors="coerce").min()
        max_weather_ts = pd.to_datetime(window["weather_time_local"], errors="coerce").max()
        window["weather_time_local"] = pd.to_datetime(
            window["weather_time_local"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "site_id": paths.site_id,
            "days": "all",
            "window_mode": "all_time",
            "start_date": pd.Timestamp(min_weather_ts).normalize().strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(max_weather_ts).normalize().strftime("%Y-%m-%d"),
            "count": int(len(window)),
            "rows": _rows_to_json_records(window),
        }

    @app.get("/get_next_day_weather_prediction")
    @app.get("/get-next-day-weather-prediction")
    def get_next_day_weather_prediction() -> list[dict[str, Any]]:
        """Return all weather rows whose local timestamp falls on tomorrow."""

        paths = get_paths(load_dotenv=True)
        weather_path = paths.gold_datasets_dir / "stuttgart_weather.csv"
        if not weather_path.exists():
            raise HTTPException(status_code=404, detail=f"Not found: {weather_path}")

        try:
            df = pd.read_csv(weather_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No rows in: {weather_path}")

        if "weather_time_local" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="stuttgart_weather.csv is missing required column: weather_time_local",
            )

        timezone = "Europe/Berlin"
        if "timezone" in df.columns:
            tz_values = df["timezone"].dropna().astype(str)
            if not tz_values.empty:
                timezone = tz_values.iloc[-1]

        df["weather_time_local"] = pd.to_datetime(df["weather_time_local"], errors="coerce")
        df = df.dropna(subset=["weather_time_local"])
        if df.empty:
            raise HTTPException(status_code=404, detail="No valid weather_time_local rows found")

        tomorrow = (
            pd.Timestamp.now(tz=timezone).tz_localize(None).normalize()
            + pd.Timedelta(days=1)
        )

        next_day_rows = df.loc[df["weather_time_local"].dt.normalize() == tomorrow].copy()
        if next_day_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No weather rows for next day: {tomorrow.strftime('%Y-%m-%d')}",
            )

        sort_cols = ["weather_time_local"]
        if "created_at_utc" in next_day_rows.columns:
            next_day_rows["created_at_utc"] = pd.to_datetime(
                next_day_rows["created_at_utc"], errors="coerce", utc=True
            )
            sort_cols.append("created_at_utc")

        next_day_rows = next_day_rows.sort_values(sort_cols)
        next_day_rows["weather_time_local"] = pd.to_datetime(
            next_day_rows["weather_time_local"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

        return _rows_to_json_records(next_day_rows)

    @app.get("/get_ho_ne_temperature_today_and_next")
    @app.get("/get-ho-ne-temperature-today-and-next")
    def get_ho_ne_temperature_today_and_next() -> list[dict[str, Any]]:
        """Return today's Ho_Ne_Temperatur and a mirrored copy with next-day timestamps."""

        paths = get_paths(load_dotenv=True)
        data_full_path = paths.gold_datasets_dir / "data_full.csv"
        if not data_full_path.exists():
            raise HTTPException(status_code=404, detail=f"Not found: {data_full_path}")

        try:
            df = pd.read_csv(data_full_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No rows in: {data_full_path}")

        if "Ho_Ne_Temperatur" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="data_full.csv is missing required column: Ho_Ne_Temperatur",
            )

        ts_col = None
        for candidate in ["zeit", "date", "datetime", "timestamp"]:
            if candidate in df.columns:
                ts_col = candidate
                break
        if ts_col is None:
            raise HTTPException(
                status_code=500,
                detail="data_full.csv is missing a supported timestamp column (zeit/date/datetime/timestamp)",
            )

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df["Ho_Ne_Temperatur"] = pd.to_numeric(df["Ho_Ne_Temperatur"], errors="coerce")
        df = df.dropna(subset=[ts_col, "Ho_Ne_Temperatur"])
        if df.empty:
            raise HTTPException(status_code=404, detail="No valid timestamp/Ho_Ne_Temperatur rows found")

        today = pd.Timestamp.now().normalize()
        today_rows = df.loc[df[ts_col].dt.normalize() == today, [ts_col, "Ho_Ne_Temperatur"]].copy()
        if today_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No Ho_Ne_Temperatur rows found for current day: {today.strftime('%Y-%m-%d')}",
            )

        today_rows = today_rows.sort_values(ts_col)
        today_rows["timestamp"] = pd.to_datetime(today_rows[ts_col], errors="coerce")

        next_rows = today_rows.copy()
        next_rows["timestamp"] = next_rows["timestamp"] + pd.Timedelta(days=1)

        out = pd.concat([today_rows, next_rows], ignore_index=True)
        out = out[["timestamp", "Ho_Ne_Temperatur"]].copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        return _rows_to_json_records(out)

    @app.get("/get_last_30d_predictions")
    @app.get("/get-last-30d-predictions")
    def get_last_30d_predictions() -> list[dict[str, Any]]:
        """Return all predictions with a minimal schema plus optional inferred next-day row."""

        paths = get_paths(load_dotenv=True)
        predictions_path = paths.gold_datasets_dir / "predictions.csv"
        if not predictions_path.exists():
            raise HTTPException(status_code=404, detail=f"Not found: {predictions_path}")

        try:
            df = pd.read_csv(predictions_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No rows in: {predictions_path}")

        out = df.copy()
        out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]

        lower_to_name = {str(c).strip().lower(): str(c) for c in out.columns}

        def _col_name(name: str) -> str | None:
            return lower_to_name.get(name.strip().lower())

        def _series_or_none(name: str) -> pd.Series | None:
            col = _col_name(name)
            if col is None:
                return None
            return out[col]

        def _parse_date_like(series: pd.Series) -> pd.Series:
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().any():
                return parsed
            return pd.to_datetime(series, errors="coerce", dayfirst=True)

        target_series = _series_or_none("target")
        if target_series is not None:
            out["target"] = target_series.apply(_normalize_prediction_target_label)
        else:
            out["target"] = None

        feature_date_series = _series_or_none("feature_date")
        if feature_date_series is not None:
            out["feature_date"] = _parse_date_like(feature_date_series).dt.strftime(
                "%Y-%m-%d"
            )
        else:
            out["feature_date"] = None

        prediction_date_series = _series_or_none("prediction_date")
        if prediction_date_series is not None:
            out["prediction_date"] = _parse_date_like(prediction_date_series).dt.strftime(
                "%Y-%m-%d"
            )
        else:
            out["prediction_date"] = None

        prediction_series = _series_or_none("prediction")
        if prediction_series is None:
            raise HTTPException(
                status_code=500,
                detail="predictions.csv is missing required column: prediction",
            )
        out["prediction"] = prediction_series

        def _row_prediction_to_bool(row: pd.Series) -> bool:
            return _prediction_value_to_bool(
                row.get("prediction"),
                target=row.get("target"),
            )

        try:
            out["prediction"] = out.apply(_row_prediction_to_bool, axis=1)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        minimal = out[["target", "feature_date", "prediction_date", "prediction"]].copy()
        minimal["prediction_date_dt"] = pd.to_datetime(minimal["prediction_date"], errors="coerce")

        def _find_col_ci(df_in: pd.DataFrame, preferred: list[str]) -> str | None:
            lower_to_actual = {str(c).strip().lower(): str(c) for c in df_in.columns}
            for token in preferred:
                hit = lower_to_actual.get(token.strip().lower())
                if hit is not None:
                    return hit
            return None

        # Business rule: if latest known day is True, infer next day True when
        # next-day noon weather is similar (temp within +/-4C) and no rain at noon.
        latest_true = minimal.loc[
            minimal["prediction"].astype(bool) & minimal["prediction_date_dt"].notna()
        ].sort_values("prediction_date_dt")

        if not latest_true.empty:
            base = latest_true.iloc[-1]
            base_day = pd.Timestamp(base["prediction_date_dt"]).normalize()
            next_day = base_day + pd.Timedelta(days=1)

            already_has_next_day = (minimal["prediction_date_dt"].dt.normalize() == next_day).any()

            if not already_has_next_day:
                weather_path = paths.gold_datasets_dir / "stuttgart_weather.csv"
                if weather_path.exists():
                    try:
                        weather = pd.read_csv(weather_path)
                        weather.columns = [str(c).replace("\ufeff", "").strip() for c in weather.columns]

                        ts_col = _find_col_ci(weather, ["weather_time_local"])
                        temp_col = _find_col_ci(weather, ["temperature_2m", "temperature", "temp_2m"])
                        rain_col = _find_col_ci(weather, ["rain", "precipitation"])

                        if ts_col is not None and temp_col is not None and rain_col is not None:
                            weather[ts_col] = pd.to_datetime(weather[ts_col], errors="coerce")
                            weather = weather.dropna(subset=[ts_col])
                            weather = weather.loc[weather[ts_col].dt.hour == 12].copy()

                            if not weather.empty:
                                weather["day"] = weather[ts_col].dt.normalize()
                                weather = weather.sort_values(ts_col)
                                noon_by_day = weather.groupby("day", as_index=False).tail(1)

                                base_noon = noon_by_day.loc[noon_by_day["day"] == base_day]
                                next_noon = noon_by_day.loc[noon_by_day["day"] == next_day]

                                if (not base_noon.empty) and (not next_noon.empty):
                                    base_temp = pd.to_numeric(base_noon[temp_col], errors="coerce").iloc[-1]
                                    next_temp = pd.to_numeric(next_noon[temp_col], errors="coerce").iloc[-1]
                                    next_rain = pd.to_numeric(next_noon[rain_col], errors="coerce").iloc[-1]

                                    similar_temp = pd.notna(base_temp) and pd.notna(next_temp) and (
                                        abs(float(next_temp) - float(base_temp)) <= 4.0
                                    )
                                    no_rain = pd.notna(next_rain) and (float(next_rain) <= 0.0)

                                    if similar_temp and no_rain:
                                        feature_date_val: str | None = None
                                        try:
                                            base_feature_date = pd.to_datetime(base.get("feature_date"), errors="coerce")
                                            if pd.notna(base_feature_date):
                                                feature_date_val = (
                                                    pd.Timestamp(base_feature_date).normalize() + pd.Timedelta(days=1)
                                                ).strftime("%Y-%m-%d")
                                        except Exception:
                                            feature_date_val = None

                                        inferred = pd.DataFrame(
                                            [
                                                {
                                                    "target": base.get("target"),
                                                    "feature_date": feature_date_val,
                                                    "prediction_date": next_day.strftime("%Y-%m-%d"),
                                                    "prediction": True,
                                                    "prediction_date_dt": next_day,
                                                }
                                            ]
                                        )
                                        minimal = pd.concat([minimal, inferred], ignore_index=True)
                    except Exception:
                        # Never fail endpoint on optional inferred-next-day enhancement.
                        pass

        minimal = minimal.sort_values("prediction_date_dt", na_position="last")
        minimal = minimal[["target", "feature_date", "prediction_date", "prediction"]].copy()
        return _rows_to_json_records(minimal)

    return app


app = create_app()
