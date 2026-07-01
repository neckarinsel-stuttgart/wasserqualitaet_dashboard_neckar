"""Bronze step: fetch and unpack DWD hourly station data.

This `.py` file is the canonical pipeline implementation.
The notebook `scripts/bronze/crawl_dwd.ipynb` is for convenience only and may drift.
"""

from __future__ import annotations

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x=None):
        print(x)

import logging
import os

def _get_log_level() -> int:
    level_name = (os.getenv('DWD_LOG_LEVEL') or 'INFO').upper().strip()
    return getattr(logging, level_name, logging.INFO)

_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'
_level = _get_log_level()
# force=True ensures notebooks re-apply root logging config on re-run
logging.basicConfig(level=_level, format=_LOG_FORMAT, force=True)
logger = logging.getLogger('dwd')
logger.setLevel(_level)
logger.info('Logger initialized')


import os
import re
import io
import glob
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # loads variables from a `.env` file in the working directory

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
DATA_DIR = resolve_path(os.getenv('BRONZE_DWD_DIR'), DATA_BRONZE_DIR / 'dwd', ROOT)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = str(DATA_DIR)

# Write logs next to the downloaded data by default
_log_file = os.getenv('DWD_LOG_FILE') or os.path.join(DATA_DIR, 'dwd_crawl.log')
try:
    # Replace any existing file handlers (re-running the cell should not duplicate logs)
    for _h in list(logger.handlers):
        if isinstance(_h, logging.FileHandler):
            logger.removeHandler(_h)
            try:
                _h.close()
            except Exception:
                pass
    _fh = logging.FileHandler(_log_file, encoding='utf-8')
    _fh.setLevel(logger.level)
    _fh.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(_fh)
    logger.info(f'Logging to file: {_log_file}')
except Exception:
    logger.exception(f'Failed to configure file logging to: {_log_file}')

BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/"
CATEGORIES = {
    "air_temperature": "lufttemp",
    "moisture": "feuchtigkeit",
    "pressure": "druck",
    "soil_temperature": "boden",
    "solar": "strahlung",
    "sun": "sun",  # sunshine duration
    "precipitation": "regen",  # keep legacy filename compatibility
}
ID_MIN = 4926
ID_MAX = 4933

# recents/ is created inside the bronze DWD folder
RECENT_DIR = os.path.join(DATA_DIR, "recents")
os.makedirs(RECENT_DIR, exist_ok=True)

logger.info(f'Repo root: {ROOT}')
logger.info(f'DWD data dir: {DATA_DIR}')
logger.info(f'DWD recents dir: {RECENT_DIR}')



from datetime import datetime, timedelta, timezone
from email.utils import formatdate
import json


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def pick_timestamp_column(df: pd.DataFrame) -> str | None:
    if "MESS_DATUM" in df.columns:
        return "MESS_DATUM"
    for alt in ["MESS_DATUM_BEGINN", "MESS_DATUM_BEG", "MESS_DATUM_ANFANG"]:
        if alt in df.columns:
            return alt
    return None


def fix_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df.copy())
    ts_col = pick_timestamp_column(df)
    if ts_col is None:
        return df

    # DWD hourly timestamps are typically YYYYMMDDHH (10 chars)
    df["DATUM"] = df[ts_col].astype(str).str.slice(0, 10)
    df["DATUM"] = pd.to_datetime(df["DATUM"], format="%Y%m%d%H", errors="coerce")
    df.set_index("DATUM", inplace=True)
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    renames = {}
    return df.rename(columns=renames)


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        c
        for c in [
            "EOR",
            "STATIONS_ID",
            "MESS_DATUM",
            "MESS_DATUM_BEGINN",
            "MESS_DATUM_BEG",
            "MESS_DATUM_ANFANG",
        ]
        if c in df.columns
    ]
    return df.drop(columns=cols_to_drop, errors="ignore")


def _env_flag(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if v == "":
        return default
    return v in {"1", "true", "yes", "y", "on"}


def _fmt_local_ts(epoch_seconds: float) -> str:
    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds")
    )


def _rfc1123_from_epoch(epoch_seconds: float) -> str:
    return formatdate(epoch_seconds, usegmt=True)


# Behavior toggles
# - DWD_CHECK_REMOTE: by default ON so we don't silently keep stale *_akt.zip files forever.
# - DWD_FORCE_DOWNLOAD: always download, ignoring conditional caching headers.
DWD_FORCE_DOWNLOAD = _env_flag("DWD_FORCE_DOWNLOAD", False)
DWD_CHECK_REMOTE = _env_flag("DWD_CHECK_REMOTE", True)
DWD_WRITE_META = _env_flag("DWD_WRITE_META", True)
DWD_DEBUG_MAX_FILES = int(os.getenv("DWD_DEBUG_MAX_FILES") or "8")
REQUEST_HEADERS = {"User-Agent": os.getenv("DWD_USER_AGENT") or "NI_AI DWD crawler"}


def _meta_path(dest_path: str) -> str:
    return dest_path + ".meta.json"


def _load_meta(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.debug(f"Failed to read meta file: {path}")
        return {}


def _save_meta(path: str, meta: dict) -> None:
    if not DWD_WRITE_META:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.debug(f"Failed to write meta file: {path}")


def fetch_zip(url: str, dest_path: str) -> tuple[bytes, str]:
    """Fetch a DWD zip.

    Returns (content, status) where status is one of: downloaded | not_modified | cached

    - If DWD_CHECK_REMOTE is enabled, uses conditional GET (ETag/Last-Modified or local mtime)
      to refresh files only when the remote changes.
    """

    meta_file = _meta_path(dest_path)
    meta = _load_meta(meta_file)

    req_headers = dict(REQUEST_HEADERS)

    # Build conditional headers (only when checking remote and not forcing download)
    if DWD_CHECK_REMOTE and not DWD_FORCE_DOWNLOAD and os.path.exists(dest_path):
        if meta.get("etag"):
            req_headers["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"):
            req_headers["If-Modified-Since"] = meta["last_modified"]
        else:
            try:
                st = os.stat(dest_path)
                req_headers["If-Modified-Since"] = _rfc1123_from_epoch(st.st_mtime)
            except Exception:
                pass

    logger.debug(f"Downloading: {url}")
    resp = requests.get(url, timeout=60, allow_redirects=True, headers=req_headers)

    logger.info(
        "HTTP GET %s -> %s final=%s content_length=%s last_modified=%s etag=%s (if_none_match=%s if_modified_since=%s)",
        url,
        resp.status_code,
        resp.url,
        resp.headers.get("Content-Length"),
        resp.headers.get("Last-Modified"),
        resp.headers.get("ETag"),
        req_headers.get("If-None-Match"),
        req_headers.get("If-Modified-Since"),
    )

    if resp.status_code == 304 and os.path.exists(dest_path):
        st = os.stat(dest_path)
        logger.info(
            "Not modified (304) -> using cached file: %s size=%s mtime=%s",
            dest_path,
            st.st_size,
            _fmt_local_ts(st.st_mtime),
        )
        with open(dest_path, "rb") as f:
            return f.read(), "not_modified"

    resp.raise_for_status()

    content = resp.content
    logger.debug(f"Downloaded {len(content)} bytes from: {resp.url}")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as fout:
        fout.write(content)

    # Persist remote metadata for next conditional request
    new_meta = {
        "url": url,
        "final_url": resp.url,
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "etag": resp.headers.get("ETag"),
        "last_modified": resp.headers.get("Last-Modified"),
        "content_length": resp.headers.get("Content-Length"),
    }
    meta.update({k: v for k, v in new_meta.items() if v is not None})
    _save_meta(meta_file, meta)

    return content, "downloaded"


def head_file(url: str) -> dict[str, str]:
    try:
        resp = requests.head(url, timeout=30, allow_redirects=True, headers=REQUEST_HEADERS)
        logger.info(
            "HTTP HEAD %s -> %s final=%s content_length=%s last_modified=%s etag=%s",
            url,
            resp.status_code,
            resp.url,
            resp.headers.get("Content-Length"),
            resp.headers.get("Last-Modified"),
            resp.headers.get("ETag"),
        )
        resp.raise_for_status()
        return dict(resp.headers)
    except Exception:
        logger.exception(f"HEAD failed (continuing without remote metadata): {url}")
        return {}


def list_files(url: str) -> list[str]:
    logger.debug(f"Listing files from: {url}")
    resp = requests.get(url, timeout=60, allow_redirects=True, headers=REQUEST_HEADERS)
    logger.info(
        "HTTP LIST %s -> %s final=%s html_bytes=%s",
        url,
        resp.status_code,
        resp.url,
        len(resp.content),
    )
    resp.raise_for_status()

    # Support both single and double quotes; DWD listings are simple HTML.
    pattern = re.compile(r"href=['\"]([^'\"]+\.zip)['\"]", re.IGNORECASE)
    files = pattern.findall(resp.text)

    # Normalize to basenames; some listings include full paths
    files = [os.path.basename(f) for f in files]
    files = sorted(set(files))

    logger.info(f"Found {len(files)} zip links at: {resp.url}")
    if files:
        logger.info(f"Zip link examples: {files[: min(DWD_DEBUG_MAX_FILES, len(files))]}")
        akt = [f for f in files if "_akt" in f.lower()]
        hist = [f for f in files if "hist" in f.lower()]
        logger.info(f"Name hints: akt={len(akt)} hist={len(hist)}")

    return files


def extract_id(name: str) -> int | None:
    m = re.search(r"_(\d{5})_", name)
    if not m:
        return None
    return int(m.group(1))


def read_zip(content: bytes) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            # pick the single data file that starts with "produkt"
            txt_files = [f for f in z.namelist() if f.lower().endswith(".txt")]
            produkt_files = [
                f for f in txt_files if os.path.basename(f).lower().startswith("produkt")
            ]
            if not produkt_files:
                logger.debug("Zip contains no produkt*.txt file")
                logger.debug(f"Zip txt files: {txt_files[:10]}")
                return pd.DataFrame()
            chosen = sorted(produkt_files)[0]
            logger.debug(f"Reading produkt file from zip: {chosen}")
            with z.open(chosen) as f:
                df = pd.read_csv(f, sep=";", encoding="latin1", dtype=str)
                return standardize_columns(df)
    except Exception:
        logger.exception("Failed to read zip content")
        return pd.DataFrame()


def _ensure_datetime_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def _collapse_duplicates_by_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure exactly one row per timestamp (index).

    For duplicates, keeps the last non-null value per column.
    This prevents pandas merge() from creating cartesian products.
    """

    if df.index.is_unique:
        return df

    dup_rows = int(df.index.duplicated(keep=False).sum())
    logger.warning(
        "Duplicate timestamps detected in clean category data: duplicate_rows=%s unique_ts=%s total_rows=%s",
        dup_rows,
        df.index.nunique(dropna=False),
        len(df),
    )

    # Sort for deterministic 'last'
    df = df.sort_index()

    # Group by timestamp and take last non-null per column
    def _last_non_null(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[-1] if len(s2) else pd.NA

    collapsed = df.groupby(level=0, sort=True).agg(_last_non_null)
    return collapsed


def get_aggregation_end() -> pd.Timestamp:
    """Return 23:00 tomorrow in local time for a complete next-day weather grid."""

    tomorrow = datetime.now().astimezone().date() + timedelta(days=1)
    return pd.Timestamp(tomorrow) + pd.Timedelta(hours=23)


def _sort_by_available_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    sort_cols = [c for c in columns if c in df.columns]
    if not sort_cols:
        return df
    return df.sort_values(sort_cols).reset_index(drop=True)


# -----------------
# Core update & aggregate
# -----------------
def update_category(cat: str, short: str) -> None:
    # 'solar' usually has no 'recent' dir on DWD; fall back to folder root and filter *_row.zip
    url = urljoin(BASE_URL, f"{cat}/recent/")
    use_row_only = False
    alt_url = urljoin(BASE_URL, f"{cat}/")
    logger.info(f"Category start: {cat} ({short})")
    logger.info(f"Primary URL: {url}")
    logger.info(f"Download flags: force={DWD_FORCE_DOWNLOAD} check_remote={DWD_CHECK_REMOTE}")

    try:
        files = list_files(url)
        if not files:
            files = list_files(alt_url)
            url = alt_url
            use_row_only = True
    except Exception:
        try:
            files = list_files(alt_url)
            url = alt_url
            use_row_only = True
        except Exception:
            logger.exception(f"Failed to list files for category: {cat}")
            return

    logger.info(f"Using URL: {url}")
    logger.info(f"Zip candidates: {len(files)} (row_only={use_row_only})")

    target_dir = os.path.join(RECENT_DIR, cat)
    os.makedirs(target_dir, exist_ok=True)

    all_frames = []
    skipped_row = 0
    skipped_id = 0
    skipped_other = 0
    downloaded = 0
    cached = 0
    not_modified = 0

    for file in files:
        if use_row_only and not file.endswith("_row.zip"):
            skipped_row += 1
            continue
        sid = extract_id(file)
        if sid is None or sid < ID_MIN or sid > ID_MAX:
            skipped_id += 1
            continue

        dest_path = os.path.join(target_dir, file)
        full_url = urljoin(url, file)

        try:
            if DWD_FORCE_DOWNLOAD or DWD_CHECK_REMOTE or not os.path.exists(dest_path):
                content, status = fetch_zip(full_url, dest_path)
                if status == "downloaded":
                    downloaded += 1
                elif status == "not_modified":
                    not_modified += 1
                else:
                    cached += 1
            else:
                st = os.stat(dest_path)
                logger.info(
                    "Cache hit (remote check off): %s size=%s mtime=%s (no download)",
                    dest_path,
                    st.st_size,
                    _fmt_local_ts(st.st_mtime),
                )
                with open(dest_path, "rb") as f:
                    content = f.read()
                cached += 1
        except Exception:
            logger.exception(f"Failed to fetch/read zip: {file}")
            skipped_other += 1
            continue

        df = read_zip(content)
        if not df.empty:
            all_frames.append(df)

    logger.info(
        "Selection stats: total=%s downloaded=%s not_modified=%s cached=%s skipped_row=%s skipped_id=%s other_errors=%s",
        len(files),
        downloaded,
        not_modified,
        cached,
        skipped_row,
        skipped_id,
        skipped_other,
    )

    if not all_frames:
        logger.warning(f"No data frames collected for category: {cat} ({short})")
        return

    df_new = pd.concat(all_frames, ignore_index=True)
    logger.info(
        f"Concatenated rows: {len(df_new):,} columns: {len(df_new.columns)} for {cat} ({short})"
    )

    # de-dup raw using available keys
    keys = [
        k
        for k in ["MESS_DATUM", "MESS_DATUM_BEGINN", "MESS_DATUM_BEG", "STATIONS_ID"]
        if k in df_new.columns
    ]
    if keys:
        df_new = df_new.drop_duplicates(subset=keys, keep="last")
        df_new = _sort_by_available_columns(df_new, keys)
        logger.info(f"De-duplicated using keys {keys}; rows now: {len(df_new):,}")

    # RAW file written into DATA_DIR (parent folder from .env)
    raw_path = os.path.join(DATA_DIR, f"schnarrenberg_dwd_{short}.csv")
    if os.path.exists(raw_path):
        df_exist = pd.read_csv(raw_path, sep=";", dtype=str)
        df_exist = standardize_columns(df_exist)
        df_new = pd.concat([df_exist, df_new], ignore_index=True)
        keys = [
            k
            for k in ["MESS_DATUM", "MESS_DATUM_BEGINN", "MESS_DATUM_BEG", "STATIONS_ID"]
            if k in df_new.columns
        ]
        if keys:
            df_new.drop_duplicates(subset=keys, keep="last", inplace=True)
            df_new = _sort_by_available_columns(df_new, keys)
    df_new.to_csv(raw_path, sep=";", index=False)
    logger.info(f"Wrote raw CSV: {raw_path} (rows={len(df_new):,})")

    # CLEAN file also in DATA_DIR (NOT in recents/)
    df_clean = fix_dates(df_new.copy())
    df_clean = rename_columns(df_clean)
    df_clean = remove_duplicate_columns(df_clean)

    # Convert index to column and normalize type to datetime for stable de-dup across runs
    df_clean = df_clean.reset_index()
    df_clean = _ensure_datetime_column(df_clean, "DATUM")

    clean_path = os.path.join(DATA_DIR, f"clean_schnarrenberg_dwd_{short}.csv")
    if os.path.exists(clean_path):
        df_exist = pd.read_csv(clean_path)
        df_exist = _ensure_datetime_column(df_exist, "DATUM")
        df_clean = pd.concat([df_exist, df_clean], ignore_index=True)

    if "DATUM" in df_clean.columns:
        before = len(df_clean)
        df_clean.sort_values("DATUM", inplace=True)
        df_clean.drop_duplicates(subset="DATUM", keep="last", inplace=True)
        df_clean.reset_index(drop=True, inplace=True)
        logger.info(
            "Clean de-dup by DATUM: before=%s after=%s removed=%s",
            before,
            len(df_clean),
            before - len(df_clean),
        )

    df_clean.to_csv(clean_path, index=False)
    logger.info(f"Wrote clean CSV: {clean_path} (rows={len(df_clean):,})")


def aggregate_all() -> None:
    files = glob.glob(os.path.join(DATA_DIR, "clean_schnarrenberg_dwd_*.csv"))
    aggregation_end = get_aggregation_end()
    date_range = pd.date_range("2023-01-01", aggregation_end, freq="h")
    df_agg = pd.DataFrame(index=date_range)
    logger.info(f"Aggregate start. Clean files found: {len(files)}")
    logger.info(
        f"Aggregate target range: {date_range.min()} to {date_range.max()} (hours={len(date_range):,})"
    )

    for file in files:
        logger.info(f"Reading clean file: {file}")
        try:
            df_part = pd.read_csv(file)
        except Exception:
            logger.exception(f"Failed to read clean file: {file}")
            continue

        if "DATUM" not in df_part.columns:
            logger.warning(f"Skipping file without DATUM column: {file}")
            continue

        df_part["DATUM"] = pd.to_datetime(df_part["DATUM"], errors="coerce")
        df_part.dropna(subset=["DATUM"], inplace=True)
        df_part.set_index("DATUM", inplace=True)

        # CRITICAL: ensure one row per timestamp before merge (prevents row explosion)
        df_part = _collapse_duplicates_by_index(df_part)

        # Align to the target hourly range to ensure stable output shape
        df_part = df_part.reindex(date_range)

        df_part = remove_duplicate_columns(df_part)

        df_agg = df_agg.merge(
            df_part,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=("", "_y"),
        )

    # Keep exactly one row per hour in the target range
    df_agg = df_agg[~df_agg.index.duplicated(keep="first")]

    out_path = os.path.join(DATA_DIR, "clean_wetter_komplett.csv")
    df_agg.to_csv(out_path)
    logger.info(
        f"Wrote aggregated weather: {out_path} (rows={len(df_agg):,} cols={len(df_agg.columns)})"
    )


def data_quality_monitoring() -> None:
    """Analyze data quality of the crawled and aggregated weather data."""

    logger.info("============================================================")
    logger.info("DATA QUALITY MONITORING REPORT")
    logger.info("============================================================")

    agg_file = os.path.join(DATA_DIR, "clean_wetter_komplett.csv")
    if not os.path.exists(agg_file):
        logger.error("Aggregated file 'clean_wetter_komplett.csv' not found")
        return

    try:
        df_agg = pd.read_csv(agg_file, index_col=0, parse_dates=True)
        logger.info("Successfully loaded aggregated data")
        logger.info(f"Dataset shape: {df_agg.shape}")
        logger.info(f"Date range: {df_agg.index.min()} to {df_agg.index.max()}")
        logger.info(f"Total hours covered: {len(df_agg):,}")
    except Exception as e:
        logger.exception(f"Error loading aggregated file: {e}")
        return

    total_cells = df_agg.shape[0] * df_agg.shape[1]
    null_cells = df_agg.isnull().sum().sum()
    if total_cells == 0:
        logger.warning(
            "Aggregated dataset is empty (0 rows or 0 columns). Skipping detailed quality checks."
        )
        return

    data_completeness = ((total_cells - null_cells) / total_cells) * 100

    logger.info("OVERALL DATA QUALITY")
    logger.info(f"Total data points: {total_cells:,}")
    logger.info(f"Missing data points: {null_cells:,}")
    logger.info(f"Data completeness: {data_completeness:.2f}%")

    logger.info("COLUMN-WISE ANALYSIS")
    column_stats = []

    for col in df_agg.columns:
        total_values = len(df_agg)
        missing_values = df_agg[col].isnull().sum()
        completeness = ((total_values - missing_values) / total_values) * 100

        if df_agg[col].dtype in ["float64", "int64"]:
            try:
                valid_data = df_agg[col].dropna()
                if len(valid_data) > 0:
                    min_val = valid_data.min()
                    max_val = valid_data.max()
                    mean_val = valid_data.mean()
                    zeros = (valid_data == 0).sum()
                    negatives = (valid_data < 0).sum()
                else:
                    min_val = max_val = mean_val = zeros = negatives = 0
            except Exception:
                min_val = max_val = mean_val = zeros = negatives = 0
        else:
            min_val = max_val = mean_val = zeros = negatives = None

        column_stats.append(
            {
                "column": col,
                "total": total_values,
                "missing": missing_values,
                "completeness": completeness,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "zeros": zeros,
                "negatives": negatives,
            }
        )

    column_stats.sort(key=lambda x: x["completeness"], reverse=True)

    logger.info("BEST DATA QUALITY (Top 5)")
    for i, stats in enumerate(column_stats[:5]):
        logger.info(
            f"{i+1}. {stats['column']:<25} | {stats['completeness']:6.2f}% complete | {stats['missing']:,} missing"
        )

    logger.info("WORST DATA QUALITY (Bottom 5)")
    for i, stats in enumerate(column_stats[-5:]):
        rank = len(column_stats) - 4 + i
        logger.info(
            f"{rank}. {stats['column']:<25} | {stats['completeness']:6.2f}% complete | {stats['missing']:,} missing"
        )

    excellent = [s for s in column_stats if s["completeness"] >= 95]
    good = [s for s in column_stats if 80 <= s["completeness"] < 95]
    moderate = [s for s in column_stats if 50 <= s["completeness"] < 80]
    poor = [s for s in column_stats if s["completeness"] < 50]

    logger.info("QUALITY DISTRIBUTION")
    logger.info(f"Excellent (>=95%): {len(excellent)} columns")
    logger.info(f"Good (80-95%): {len(good)} columns")
    logger.info(f"Moderate (50-80%): {len(moderate)} columns")
    logger.info(f"Poor (<50%): {len(poor)} columns")

    if poor:
        logger.warning("POOR QUALITY COLUMNS")
        for stats in poor:
            logger.warning(f"{stats['column']}: {stats['completeness']:.1f}% complete")

    logger.info("DATE COVERAGE ANALYSIS")
    expected_start = pd.Timestamp("2023-01-01")
    expected_end = get_aggregation_end()
    actual_start = df_agg.index.min()
    actual_end = df_agg.index.max()
    if pd.isna(actual_start) or pd.isna(actual_end):
        logger.warning(
            "Aggregated dataset has no valid datetime index range. Skipping date coverage analysis."
        )
        return

    expected_hours = pd.date_range(expected_start, expected_end, freq="h")
    actual_hours = len(df_agg)
    coverage_ratio = (actual_hours / len(expected_hours)) * 100

    logger.info(f"Expected range: {expected_start} to {expected_end}")
    logger.info(f"Actual range: {actual_start} to {actual_end}")
    logger.info(f"Expected hours: {len(expected_hours):,}")
    logger.info(f"Actual hours: {actual_hours:,}")
    logger.info(f"Coverage ratio: {coverage_ratio:.2f}%")

    full_range = pd.date_range(actual_start, actual_end, freq="h")
    missing_timestamps = set(full_range) - set(df_agg.index)

    if missing_timestamps:
        logger.warning(f"Missing timestamps: {len(missing_timestamps):,}")
        if len(missing_timestamps) <= 10:
            logger.warning(f"Missing dates: {sorted(list(missing_timestamps))}")
    else:
        logger.info("No missing timestamps in range")

    logger.info("CATEGORY-WISE ANALYSIS")
    categories = {
        "Temperature": [
            col
            for col in df_agg.columns
            if any(temp in col.lower() for temp in ["temp", "lufttemp"])
        ],
        "Humidity": [
            col
            for col in df_agg.columns
            if any(hum in col.lower() for hum in ["feuchte", "humidity", "rel"])
        ],
        "Pressure": [
            col
            for col in df_agg.columns
            if any(press in col.lower() for press in ["druck", "press"])
        ],
        "Solar/Sun": [
            col
            for col in df_agg.columns
            if any(sol in col.lower() for sol in ["strahlung", "solar", "sun"])
        ],
        "Soil": [
            col
            for col in df_agg.columns
            if any(soil in col.lower() for soil in ["boden", "soil"])
        ],
        "Other": [],
    }

    categorized_cols = set()
    for cat_cols in categories.values():
        categorized_cols.update(cat_cols)
    categories["Other"] = [col for col in df_agg.columns if col not in categorized_cols]

    for cat_name, cat_cols in categories.items():
        if cat_cols:
            cat_stats = [s for s in column_stats if s["column"] in cat_cols]
            avg_completeness = sum(s["completeness"] for s in cat_stats) / len(cat_stats)
            logger.info(
                f"{cat_name:<12}: {len(cat_cols):2d} columns, {avg_completeness:6.2f}% avg completeness"
            )

    logger.info("RECOMMENDATIONS")

    if data_completeness >= 90:
        logger.info(f"Excellent overall data quality ({data_completeness:.1f}%)")
    elif data_completeness >= 75:
        logger.info(
            f"Good data quality ({data_completeness:.1f}%) - monitor poor columns"
        )
    else:
        logger.warning(
            f"Poor data quality ({data_completeness:.1f}%) - investigate data sources"
        )

    if poor:
        logger.warning(f"Consider removing {len(poor)} columns with <50% completeness")

    if len(missing_timestamps) > len(full_range) * 0.05:
        logger.warning("Significant time gaps detected - check data collection continuity")

    if coverage_ratio < 80:
        logger.warning(
            f"Low temporal coverage ({coverage_ratio:.1f}%) - extend date range or increase frequency"
        )

    report_file = os.path.join(DATA_DIR, "data_quality_report.csv")
    df_report = pd.DataFrame(column_stats)
    df_report.to_csv(report_file, index=False)
    logger.info(f"Detailed report saved to: {report_file}")

    logger.info("============================================================")
    logger.info("DATA QUALITY MONITORING COMPLETED")
    logger.info("============================================================")


# -----------------
if __name__ == "__main__":
    logger.info("Starting DWD data crawling...")
    
    for cat, short in CATEGORIES.items():
        logger.info(f"Updating category: {cat} ({short})")
        update_category(cat, short)
    
    logger.info("Aggregating all categories...")
    aggregate_all()
    
    logger.info("Running data quality monitoring...")
    data_quality_monitoring()
    
    logger.info("DWD data crawling completed!")
