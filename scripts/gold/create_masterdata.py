"""Gold step: create masterdata outputs based on the full dataset.

This `.py` file is the canonical pipeline implementation.
The notebook `scripts/gold/create_masterdata.ipynb` is for convenience only and may drift.
"""

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x=None):
        print(x)

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / '.git').exists():
            return parent
    return cur

def resolve_path(env_value: str | None, default: Path, root: Path) -> Path:
    if env_value is None or env_value.strip() == '':
        return default
    p = Path(env_value)
    return p if p.is_absolute() else (root / p)

ROOT = find_repo_root()
DATA_GOLD_DIR = resolve_path(os.getenv('DATA_GOLD'), ROOT / 'data' / 'gold', ROOT)
GOLD_DATASETS_DIR = resolve_path(os.getenv('GOLD_DATASETS_DIR'), DATA_GOLD_DIR / 'datasets', ROOT)
GOLD_DATASETS_DIR.mkdir(parents=True, exist_ok=True)


data_full_file = GOLD_DATASETS_DIR / 'data_full.csv'
if not data_full_file.exists():
    raise FileNotFoundError(
        f"Missing {data_full_file}. Run scripts/gold/data_full.py first (or the pipeline) to generate it."
    )

df_full = pd.read_csv(data_full_file, index_col=0, parse_dates=True)
df_full = df_full[(df_full.index >= '2024-01-01')]


df_full = df_full.apply(pd.to_numeric, errors='coerce')


def sample_at_hour(df, hour):
    return df[df.index.hour == hour].resample('D').first()

daily_mean = df_full.resample('D').mean().add_suffix('_mean')
daily_min = df_full.resample('D').min().add_suffix('_min')
daily_max = df_full.resample('D').max().add_suffix('_max')
at_06 = sample_at_hour(df_full, 6).add_suffix('_06')
at_12 = sample_at_hour(df_full, 12).add_suffix('_12')
at_18 = sample_at_hour(df_full, 18).add_suffix('_18')
at_00 = sample_at_hour(df_full, 0).add_suffix('_00')

df_daily = pd.concat([daily_mean, daily_min, daily_max, at_06, at_12, at_18, at_00], axis=1)
df_daily.sort_index(inplace=True)


df_daily = df_daily.groupby(df_daily.index.date).apply(lambda g: g.bfill().ffill())


df_daily.to_csv(GOLD_DATASETS_DIR / 'masterdata.csv')
