"""Gold step: assemble the full modeling dataset from Silver + LUBW sources.

This `.py` file is the canonical pipeline implementation.
The notebook `scripts/gold/data_full.ipynb` is for convenience only and may drift.
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
DATA_BRONZE_DIR = resolve_path(os.getenv('DATA_BRONZE'), ROOT / 'data' / 'bronze', ROOT)
DATA_SILVER_DIR = resolve_path(os.getenv('DATA_SILVER'), ROOT / 'data' / 'silver', ROOT)
DATA_GOLD_DIR = resolve_path(os.getenv('DATA_GOLD'), ROOT / 'data' / 'gold', ROOT)

SILVER_WEATHER_DIR = resolve_path(os.getenv('SILVER_WEATHER_DIR'), DATA_SILVER_DIR / 'weather', ROOT)
SILVER_MESSUNGEN_DIR = resolve_path(os.getenv('SILVER_MESSUNGEN_DIR'), DATA_SILVER_DIR / 'messungen', ROOT)
BRONZE_LUBW_DIR = resolve_path(os.getenv('BRONZE_LUBW_DIR'), DATA_BRONZE_DIR / 'lubw', ROOT)
GOLD_DATASETS_DIR = resolve_path(os.getenv('GOLD_DATASETS_DIR'), DATA_GOLD_DIR / 'datasets', ROOT)
GOLD_DATASETS_DIR.mkdir(parents=True, exist_ok=True)


# Wetterdaten und E. coli-Messungen laden
weather_file = SILVER_WEATHER_DIR / 'clean_wetter_komplett.csv'
if not weather_file.exists():
    raise FileNotFoundError(
        f"Missing {weather_file}. Run scripts/silver/clean_dwd_daten.py (or the pipeline) first."
    )

messungen_file = SILVER_MESSUNGEN_DIR / 'messungen_komplett.csv'
if not messungen_file.exists():
    raise FileNotFoundError(
        f"Missing {messungen_file}. Run scripts/silver/create_messungen_complete.py (or the pipeline) first."
    )

df_wetter = pd.read_csv(weather_file, header=0, index_col=0, parse_dates=True)
df_messungen = pd.read_csv(messungen_file, header=0)
df_messungen['datum'] = pd.to_datetime(df_messungen['datum'], errors='coerce')
df_messungen.set_index('datum', inplace=True)


df_komplett = pd.merge(df_wetter, df_messungen, how="left", left_index=True, right_index=True)
df_komplett.index.rename("zeit", inplace=True)


df_komplett = df_komplett[df_komplett.index >= "2024-01-01"]


# LUBW-Daten laden und verarbeiten
lubw_file = BRONZE_LUBW_DIR / 'lubw_download_latest.csv'
if not lubw_file.exists():
    raise FileNotFoundError(
        f"Missing {lubw_file}. Run scripts/bronze/crawl_lubw.py (or the pipeline) first."
    )

df_lubw_raw = pd.read_csv(lubw_file, sep=';', parse_dates=['Datum'], dayfirst=True)

water_col = None
for candidate in ['Gewaesser', 'Gewässer']:
    if candidate in df_lubw_raw.columns:
        water_col = candidate
        break
if water_col is None:
    raise KeyError(f"Missing water body column in LUBW file. Columns: {list(df_lubw_raw.columns)}")

df_lubw = df_lubw_raw[['Messstation', water_col, 'Parameter', 'Datum', 'Tagesmittelwert']].copy()
if water_col != 'Gewaesser':
    df_lubw.rename(columns={water_col: 'Gewaesser'}, inplace=True)

df_lubw['Tagesmittelwert'] = df_lubw['Tagesmittelwert'].astype(str).str.replace(',', '.', regex=False)
df_lubw['Tagesmittelwert'] = pd.to_numeric(df_lubw['Tagesmittelwert'], errors='coerce')


df_lubw_raw["Datum"] = df_lubw_raw["Datum"].apply(pd.to_datetime)


# Zahlen sicher numerisch halten
df_lubw["Tagesmittelwert"] = pd.to_numeric(df_lubw["Tagesmittelwert"], errors="coerce")


# Kuerzel erstellen z.B. We_Ne_Temperatur
df_lubw["Kuerzel"] = (
    df_lubw["Messstation"].str[:2].str.capitalize() + "_" +
    df_lubw["Gewaesser"].str[:2].str.capitalize() + "_" +
    df_lubw["Parameter"]
        .str.replace("bei .*", "", regex=True)
        .str.replace(" ", "")
        .str.replace("ä", "ae")
        .str.replace("ö", "oe")
        .str.replace("ü", "ue")
        .str.replace("ß", "ss")
)


df_lubw["Kuerzel"].unique()


# Pivotieren: Datum als Index, Parameter als Spalten
df_lubw_pivot = df_lubw.pivot(index="Datum", columns="Kuerzel", values="Tagesmittelwert")

# Resample auf 1h und auffuellen
df_lubw_hourly = df_lubw_pivot.resample("1h").ffill()

# In Masterdaten integrieren
df_komplett = df_komplett.merge(df_lubw_hourly, left_index=True, right_index=True)


df_lubw_raw["Tagesmittelwert"] = pd.to_numeric(
    df_lubw_raw["Tagesmittelwert"]
    .str.replace(",", "."),
    errors="coerce"
)


df_lubw_raw["Parameter"].unique()


# Formatierung
df_komplett.columns = [col.strip() for col in df_komplett.columns]
df_komplett.index.name = 'zeit'


# Export
df_komplett.to_csv(GOLD_DATASETS_DIR / 'data_full.csv')
