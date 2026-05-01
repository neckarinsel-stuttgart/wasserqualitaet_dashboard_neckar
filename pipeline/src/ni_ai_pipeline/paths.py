from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return cur


def resolve_path(env_value: str | None, default: Path, root: Path) -> Path:
    if env_value is None or env_value.strip() == "":
        return default
    path_value = Path(env_value)
    return path_value if path_value.is_absolute() else (root / path_value)


@dataclass(frozen=True)
class PathConfig:
    root: Path

    data_bronze: Path
    data_silver: Path
    data_gold: Path

    bronze_dwd_dir: Path
    bronze_lubw_dir: Path
    bronze_messungen_dir: Path

    silver_weather_dir: Path
    silver_messungen_dir: Path

    gold_datasets_dir: Path

    site_id: str


def load_env_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # Optional dependency at runtime; in production we rely on real env vars.
        return


def get_paths(*, load_dotenv: bool = True, start: Path | None = None) -> PathConfig:
    if load_dotenv:
        load_env_if_available()

    root = find_repo_root(start=start)

    data_bronze = resolve_path(os.getenv("DATA_BRONZE"), root / "data" / "bronze", root)
    data_silver = resolve_path(os.getenv("DATA_SILVER"), root / "data" / "silver", root)
    data_gold = resolve_path(os.getenv("DATA_GOLD"), root / "data" / "gold", root)

    bronze_dwd_dir = resolve_path(os.getenv("BRONZE_DWD_DIR"), data_bronze / "dwd", root)
    bronze_lubw_dir = resolve_path(os.getenv("BRONZE_LUBW_DIR"), data_bronze / "lubw", root)
    bronze_messungen_dir = resolve_path(
        os.getenv("BRONZE_MESSUNGEN_DIR"), data_bronze / "messungen", root
    )

    silver_weather_dir = resolve_path(
        os.getenv("SILVER_WEATHER_DIR"), data_silver / "weather", root
    )
    silver_messungen_dir = resolve_path(
        os.getenv("SILVER_MESSUNGEN_DIR"), data_silver / "messungen", root
    )

    gold_datasets_dir = resolve_path(
        os.getenv("GOLD_DATASETS_DIR"), data_gold / "datasets", root
    )

    site_id = (os.getenv("SITE_ID") or "default").strip()

    return PathConfig(
        root=root,
        data_bronze=data_bronze,
        data_silver=data_silver,
        data_gold=data_gold,
        bronze_dwd_dir=bronze_dwd_dir,
        bronze_lubw_dir=bronze_lubw_dir,
        bronze_messungen_dir=bronze_messungen_dir,
        silver_weather_dir=silver_weather_dir,
        silver_messungen_dir=silver_messungen_dir,
        gold_datasets_dir=gold_datasets_dir,
        site_id=site_id,
    )
