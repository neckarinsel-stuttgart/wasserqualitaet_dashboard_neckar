"""Silver step: build combined measurements table (Messungen) across years.

This `.py` file is the canonical pipeline implementation.
The notebook `scripts/silver/create_messungen_complete.ipynb` is for convenience only and may drift.
"""

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x=None):
        print(x)

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / 'databricks.yml').exists() or (parent / '.git').exists():
            return parent
    return cur


def resolve_path(env_value: str | None, default: Path, root: Path) -> Path:
    if env_value is None or env_value.strip() == '':
        return default
    p = Path(env_value)
    return p if p.is_absolute() else (root / p)


ROOT = find_repo_root()
DATA_BRONZE_DIR = resolve_path(os.getenv('DATA_BRONZE'), ROOT / 'data' / 'bronze', ROOT)
DATA_SILVER_DIR = resolve_path(os.getenv('DATA_SILVER'), ROOT / 'data' / 'silver', ROOT)
BRONZE_MESSUNGEN_DIR = resolve_path(
    os.getenv('BRONZE_MESSUNGEN_DIR'), DATA_BRONZE_DIR / 'messungen', ROOT
)
SILVER_MESSUNGEN_DIR = resolve_path(
    os.getenv('SILVER_MESSUNGEN_DIR'), DATA_SILVER_DIR / 'messungen', ROOT
)

SILVER_MESSUNGEN_DIR.mkdir(parents=True, exist_ok=True)
print(f'BRONZE_MESSUNGEN_DIR={BRONZE_MESSUNGEN_DIR}')
print(f'SILVER_MESSUNGEN_DIR={SILVER_MESSUNGEN_DIR}')


def _existing_messungen_years() -> list[int]:
    years: list[int] = []
    for year in (2024, 2025):
        if (BRONZE_MESSUNGEN_DIR / f'messungen_{year}.csv').exists():
            years.append(year)
    return years


years = _existing_messungen_years()
if not years:
    raise FileNotFoundError(
        f"No messungen_*.csv found in {BRONZE_MESSUNGEN_DIR}. "
        "Put files under data/bronze/messungen (or set BRONZE_MESSUNGEN_DIR/DATA_BRONZE)."
    )

print(f"Found measurement years: {years}")


def clean_value(val):
    if pd.isna(val) or val == '':
        return np.nan
    val_str = str(val).replace('>', '').replace(',', '.')
    try:
        return float(val_str)
    except Exception:
        return np.nan


frames: list[pd.DataFrame] = []

# Optional: Read 2024 data (wide format)
df_2024_clean = pd.DataFrame(columns=['datum', 'ecoli', 'entro'])
if 2024 in years:
    df_2024 = pd.read_csv(BRONZE_MESSUNGEN_DIR / 'messungen_2024.csv', sep=';', index_col=0)

    # Extract dates from column headers
    dates_2024: list[pd.Timestamp] = []
    for col in df_2024.columns:
        try:
            parts = str(col).split()
            date_str = " ".join(parts[:3])
            date_parsed = pd.to_datetime(date_str, format="%d. %b %y", errors="coerce")
            dates_2024.append(date_parsed)
        except Exception:
            dates_2024.append(pd.NaT)

    ecoli_2024 = df_2024.loc["E. Coli"].values
    entro_2024 = df_2024.loc["Enterokokken"].values

    df_2024_clean = pd.DataFrame(
        {
            'datum': dates_2024,
            'ecoli': [clean_value(v) for v in ecoli_2024],
            'entro': [clean_value(v) for v in entro_2024],
        }
    ).dropna(subset=['datum'])

    frames.append(df_2024_clean)

# Optional: Read 2025 data (already tidy)
df_2025 = pd.DataFrame(columns=['datum', 'ecoli', 'entro'])
if 2025 in years:
    df_2025 = pd.read_csv(BRONZE_MESSUNGEN_DIR / 'messungen_2025.csv', sep=';')
    df_2025['datum'] = pd.to_datetime(df_2025['DATUM'], format="%d.%m.%Y")
    df_2025 = df_2025[['datum', 'ecoli', 'entro']]

    frames.append(df_2025)

# Combine available years
if frames:
    df_combined = pd.concat(frames, ignore_index=True)
else:
    df_combined = pd.DataFrame(columns=['datum', 'ecoli', 'entro'])

df_combined = df_combined.sort_values('datum').reset_index(drop=True)

print(f"Total measurements: {len(df_combined)}")
print(f"2024 measurements: {len(df_2024_clean)}")
print(f"2025 measurements: {len(df_2025)}")
display(df_combined.head(10))
display(df_combined.tail(10))

# Save combined data
output_path = SILVER_MESSUNGEN_DIR / 'messungen_komplett.csv'
df_combined.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
