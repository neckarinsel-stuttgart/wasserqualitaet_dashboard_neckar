"""Silver step: clean DWD Bronze CSVs and build a merged hourly weather table.

This `.py` file is the canonical pipeline implementation.
The notebook `scripts/silver/clean_dwd_daten.ipynb` is for convenience only and may drift.
"""

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x=None):
        print(x)

import glob
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
DATA_BRONZE_DIR = resolve_path(os.getenv('DATA_BRONZE'), ROOT / 'data' / 'bronze', ROOT)
DATA_SILVER_DIR = resolve_path(os.getenv('DATA_SILVER'), ROOT / 'data' / 'silver', ROOT)
BRONZE_DWD_DIR = resolve_path(os.getenv('BRONZE_DWD_DIR'), DATA_BRONZE_DIR / 'dwd', ROOT)
SILVER_WEATHER_DIR = resolve_path(os.getenv('SILVER_WEATHER_DIR'), DATA_SILVER_DIR / 'weather', ROOT)
SILVER_WEATHER_DIR.mkdir(parents=True, exist_ok=True)

if not list(BRONZE_DWD_DIR.glob('schnarrenberg_dwd*.csv')):
    raise FileNotFoundError(
        f"No DWD CSVs found in {BRONZE_DWD_DIR}. "
        "Run scripts/bronze/crawl_dwd.py first or set BRONZE_DWD_DIR/DATA_BRONZE."
    )

print(f'BRONZE_DWD_DIR={BRONZE_DWD_DIR}')
print(f'SILVER_WEATHER_DIR={SILVER_WEATHER_DIR}')


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def pick_timestamp_column(df: pd.DataFrame) -> str:
    # Some inputs may already contain a parsed DATUM column.
    if 'DATUM' in df.columns:
        return 'DATUM'
    for col in ['MESS_DATUM', 'MESS_DATUM_BEGINN', 'MESS_DATUM_BEG', 'MESS_DATUM_ANFANG']:
        if col in df.columns:
            return col
    raise KeyError(f'No timestamp column found. Columns: {list(df.columns)}')

def fix_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df.copy())
    ts_col = pick_timestamp_column(df)
    if ts_col == 'DATUM':
        df['DATUM'] = pd.to_datetime(df['DATUM'], errors='coerce')
    else:
        df['DATUM'] = df[ts_col].astype(str).str.slice(0, 10)
        df['DATUM'] = pd.to_datetime(df['DATUM'], format='%Y%m%d%H', errors='coerce')
    df.set_index('DATUM', inplace=True)
    return df


def rename_columns(df):
    # Hier die gewünschten Parameternamen ergänzen
    renames = {
        
    }
    df.rename(columns = renames, inplace = True)
    return df


def create_cleans() -> None:
    all_files = glob.glob(str(BRONZE_DWD_DIR / 'schnarrenberg_dwd*.csv'))
    if not all_files:
        raise FileNotFoundError(f'No DWD files found in: {BRONZE_DWD_DIR}')

    # Skip known non-raw artifacts (already aggregated/feature files)
    all_files = [f for f in all_files if 'regen_rollups' not in Path(f).name.lower()]

    for file in all_files:
        try:
            df = pd.read_csv(file, delimiter=';', header=0, dtype=str)
            # If the file is comma-separated but we read with ';', pandas creates a single column
            # whose name contains commas (e.g. 'DATUM,R1_DAILY,...'). Detect and re-read.
            if len(df.columns) == 1 and ',' in str(df.columns[0]):
                df = pd.read_csv(file, delimiter=',', header=0)
        except Exception as e:
            print(f'Skipping unreadable file: {file} ({e})')
            continue

        # Only process raw DWD files with a timestamp column (or DATUM already present)
        df_std = standardize_columns(df.copy())
        has_ts = any(c in df_std.columns for c in ['MESS_DATUM', 'MESS_DATUM_BEGINN', 'MESS_DATUM_BEG', 'MESS_DATUM_ANFANG', 'DATUM'])
        if not has_ts:
            print(f'Skipping non-DWD-raw file (no timestamp col): {file} columns={list(df_std.columns)}')
            continue

        df = fix_dates(df)
        df = rename_columns(df)
        out_file = SILVER_WEATHER_DIR / f"clean_{Path(file).name}"
        df.to_csv(out_file)


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    cols_to_drop = [
        'EOR',
        'STATIONS_ID',
        'MESS_DATUM',
        'MESS_DATUM_BEGINN',
        'MESS_DATUM_BEG',
        'MESS_DATUM_ANFANG',
    ]
    return df.drop(columns=cols_to_drop, errors='ignore')


def create_master() -> pd.DataFrame:
    all_files = glob.glob(str(SILVER_WEATHER_DIR / 'clean_schnarrenberg_dwd*.csv'))
    if not all_files:
        raise FileNotFoundError(f'No cleaned DWD files found in: {SILVER_WEATHER_DIR}')

    # Build a continuous hourly index from fixed start to max available timestamp in cleaned files.
    max_dates = []
    prepared = []

    for file in all_files:
        df_new = pd.read_csv(file, delimiter=',', header=0)
        df_new = remove_duplicate_columns(df_new)
        df_new['DATUM'] = pd.to_datetime(df_new['DATUM'], errors='coerce')
        df_new = df_new.dropna(subset=['DATUM']).copy()
        if not df_new.empty:
            max_dates.append(df_new['DATUM'].max())
        df_new.set_index('DATUM', inplace=True)

        # Ensure one row per timestamp (avoid merge cartesian explosions)
        df_new.index = pd.to_datetime(df_new.index, errors='coerce')
        df_new = df_new[~df_new.index.isna()].copy()
        df_new.sort_index(inplace=True)
        df_new = df_new[~df_new.index.duplicated(keep='last')].copy()

        prepared.append(df_new)

    if not max_dates:
        raise ValueError('No valid DATUM values found in cleaned DWD files.')

    dynamic_end = max(max_dates).floor('h')
    date_range = pd.date_range(start='2023-01-01', end=dynamic_end, freq='h')
    df = pd.DataFrame(index=date_range)

    for df_new in prepared:
        df = pd.merge(df, df_new, how='left', left_index=True, right_index=True, suffixes=('', '_y'))

    return df


create_cleans()


merged = create_master()


merged = merged.dropna(how = "all")


merged.to_csv(SILVER_WEATHER_DIR / 'clean_wetter_komplett.csv')
