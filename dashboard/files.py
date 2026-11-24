"""Read-only access to the bronze/silver/gold folders. No Streamlit code in here."""

from __future__ import annotations

import codecs
import gzip
import io
import os
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd

LAYERS = ("bronze", "silver", "gold")
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_PREVIEW_BYTES = 1024 * 1024
MAX_ZIP_ENTRIES = 500
RAW_DIRS = {"historical", "recents"}


class FileAccessError(Exception):
    """The requested path is not allowed or cannot be read."""


@dataclass(frozen=True)
class Entry:
    name: str
    rel: str
    is_dir: bool
    size: int
    mtime: float


def data_root() -> Path:
    return Path(os.getenv("DATA_ROOT", "/data"))


def layer_root(layer: str, root: Path | None = None) -> Path:
    if layer not in LAYERS:
        raise FileAccessError("Unknown layer")
    return ((root or data_root()) / layer).resolve()


def safe_path(layer: str, rel: str = "", root: Path | None = None) -> Path:
    """Resolve rel inside the layer; reject traversal, absolute paths and outside symlinks."""
    base = layer_root(layer, root)
    if "\x00" in rel or "\\" in rel or ":" in rel:
        raise FileAccessError("Path not allowed")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts:
        raise FileAccessError("Path not allowed")
    try:
        target = (base / pure).resolve()
    except (OSError, RuntimeError) as exc:
        raise FileAccessError("Path not allowed") from exc
    if target != base and base not in target.parents:
        raise FileAccessError("Path not allowed")
    return target


def is_raw(name: str, is_dir: bool) -> bool:
    low = name.lower()
    if low.startswith("."):
        return True
    if is_dir:
        return low in RAW_DIRS
    return low.endswith((".log", ".meta.json")) or ".log." in low


def list_dir(
    layer: str, rel: str = "", show_raw: bool = False, root: Path | None = None
) -> list[Entry]:
    folder = safe_path(layer, rel, root)
    if not folder.is_dir():
        raise FileAccessError("Folder not found")
    entries: list[Entry] = []
    for child in folder.iterdir():
        child_rel = str(PurePosixPath(rel) / child.name)
        try:
            target = safe_path(layer, child_rel, root)
            stat = target.stat()
        except (FileAccessError, OSError):
            continue  # outside symlink, or removed while listing
        is_dir = target.is_dir()
        if not is_dir and not target.is_file():
            continue
        if not show_raw and is_raw(child.name, is_dir):
            continue
        entries.append(Entry(child.name, child_rel, is_dir, stat.st_size, stat.st_mtime))
    return sorted(entries, key=lambda e: (not e.is_dir, e.name.lower()))


def kind(name: str) -> str:
    low = name.lower()
    for suffix, value in (
        (".csv", "csv"),
        (".json", "json"),
        (".txt", "text"),
        (".log", "text"),
        (".zip", "zip"),
        (".gz", "gz"),
    ):
        if low.endswith(suffix):
            return value
    return "none"  # e.g. .pkl: loading a pickle can execute code, so never preview it


def file_path(layer: str, rel: str, root: Path | None = None) -> Path:
    path = safe_path(layer, rel, root)
    if not path.is_file():
        raise FileAccessError("File not found")
    return path


def read_download(layer: str, rel: str, root: Path | None = None) -> bytes:
    path = file_path(layer, rel, root)
    with path.open("rb") as f:
        data = f.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise FileAccessError("File is too large to download")
    return data


def _decode(raw: bytes, truncated: bool = False) -> str:
    """Decode bytes; a truncated chunk may end inside a multi-byte character."""
    if raw.startswith(codecs.BOM_UTF8):
        raw = raw[3:]
    for cut in range(4 if truncated else 1):
        try:
            return raw[: len(raw) - cut].decode("utf-8")
        except UnicodeDecodeError:
            continue
    return raw.decode("cp1252", errors="replace")


def detect_separator(header: str) -> str:
    counts = {sep: header.count(sep) for sep in (",", ";", "\t", "|")}
    best = max(counts, key=lambda sep: counts[sep])
    return best if counts[best] else ","


def _csv_lines(path: Path, rows: int, tail: bool) -> list[str]:
    with path.open("rb") as f:
        header = f.readline(MAX_PREVIEW_BYTES)
        if not header.strip():
            raise ValueError("empty file")
        start = f.tell()
        size = f.seek(0, os.SEEK_END)
        if tail and size - start > MAX_PREVIEW_BYTES:
            f.seek(size - MAX_PREVIEW_BYTES)
            f.readline()  # drop the partial line
        else:
            f.seek(start)
        chunk = f.read(MAX_PREVIEW_BYTES)
        truncated = f.tell() < size
    body = chunk.splitlines()
    if truncated and body:
        body.pop()  # last line may be cut off
    picked = list(deque(body, maxlen=rows)) if tail else body[:rows]
    return [_decode(line) for line in [header.rstrip(b"\r\n"), *picked] if line.strip()]


def preview_csv(
    layer: str,
    rel: str,
    rows: int = 200,
    tail: bool = False,
    root: Path | None = None,
    retries: int = 3,
    wait: float = 0.4,
) -> pd.DataFrame:
    """First/last rows as plain text. Retries because the pipeline overwrites CSVs in place."""
    path = file_path(layer, rel, root)
    error: Exception | None = None
    for attempt in range(retries):
        if attempt:
            time.sleep(wait)
        try:
            lines = _csv_lines(path, rows, tail)
            return pd.read_csv(
                io.StringIO("\n".join(lines)),
                sep=detect_separator(lines[0]),
                dtype=str,
                keep_default_na=False,
                index_col=False,
                on_bad_lines="skip",
            )
        except (OSError, ValueError) as exc:  # pandas parser errors are ValueErrors
            error = exc
    raise FileAccessError("File is empty or being updated, try again in a moment") from error


def preview_text(layer: str, rel: str, root: Path | None = None) -> tuple[str, bool]:
    with file_path(layer, rel, root).open("rb") as f:
        raw = f.read(MAX_PREVIEW_BYTES + 1)
    truncated = len(raw) > MAX_PREVIEW_BYTES
    return _decode(raw[:MAX_PREVIEW_BYTES], truncated), truncated


def preview_gz(layer: str, rel: str, root: Path | None = None) -> tuple[str, bool]:
    try:
        with gzip.open(file_path(layer, rel, root), "rb") as f:
            raw = f.read(MAX_PREVIEW_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise FileAccessError("Archive cannot be read") from exc
    truncated = len(raw) > MAX_PREVIEW_BYTES
    return _decode(raw[:MAX_PREVIEW_BYTES], truncated), truncated


def preview_zip(layer: str, rel: str, root: Path | None = None) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(file_path(layer, rel, root)) as z:
            infos = z.infolist()[:MAX_ZIP_ENTRIES]
    except (OSError, zipfile.BadZipFile) as exc:
        raise FileAccessError("Archive cannot be read") from exc
    return pd.DataFrame({"Datei": [i.filename for i in infos], "Bytes": [i.file_size for i in infos]})
