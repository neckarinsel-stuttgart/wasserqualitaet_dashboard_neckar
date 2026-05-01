from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent and not (
    (ROOT / "databricks.yml").exists() or (ROOT / ".git").exists()
):
    ROOT = ROOT.parent

PATTERNS: dict[str, re.Pattern[str]] = {
    "legacy_skripts": re.compile(r"skripts/|\\\\skripts\\\\", re.IGNORECASE),
    "legacy_daten": re.compile(r"Daten/|\\\\Daten\\\\", re.IGNORECASE),
    # NOTE: "legacy_env" is handled specially in scan() to avoid false positives
    # like "out = df.copy()" in regular Python code.
    "legacy_env": re.compile(r"path_to_data|path_to_master", re.IGNORECASE),
}

LEGACY_ENV_CODE = re.compile(
    r"path_to_data|path_to_master|os\.getenv\(\s*['\"]out['\"]\s*\)",
    re.IGNORECASE,
)

LEGACY_ENV_DOTENV = re.compile(
    r"^\s*(path_to_data|path_to_master|out)\s*=",
    re.IGNORECASE,
)

INCLUDE_SUFFIXES = {".ipynb", ".py", ".md", ".yml", ".yaml", ".env"}
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".databricks",
    ".ipynb_checkpoints",
    "papermill_outputs",
    # Generated/large trees (we don't scan data artifacts; pruning keeps scans fast)
    "data",
    # Legacy data dumps / analysis outputs can be huge; keep the notebooks, skip raw tables
    "Wetter_2024",
    "Aggregations",
    "correlations",
}

DEFAULT_SCAN_DIRS = {
    "scripts",
    "tools",
    "pipeline",
}


def _walk_files(root: Path) -> list[Path]:
    files: list[Path] = []

    # Use os.walk so we can prune directories early (Path.rglob can't do that).
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if p.name == ".env":
                files.append(p)
                continue
            if p.suffix.lower() not in INCLUDE_SUFFIXES:
                continue
            files.append(p)

    return files


def iter_files(scan_all: bool) -> list[Path]:
    # Always include top-level config/docs files.
    files: list[Path] = []
    for p in ROOT.iterdir():
        if not p.is_file():
            continue
        if p.name == ".env" or p.suffix.lower() in INCLUDE_SUFFIXES:
            files.append(p)

    if scan_all:
        files.extend(_walk_files(ROOT))
        return sorted(set(files))

    # Fast mode: scan only current pipeline surfaces.
    for name in DEFAULT_SCAN_DIRS:
        d = ROOT / name
        if d.exists() and d.is_dir():
            files.extend(_walk_files(d))

    return sorted(set(files))


def scan(scan_all: bool) -> dict[str, list[tuple[str, int, str]]]:
    hits: dict[str, list[tuple[str, int, str]]] = {k: [] for k in PATTERNS}
    for file in iter_files(scan_all=scan_all):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = text.splitlines()
        for key, pat in PATTERNS.items():
            if key == "legacy_env":
                if file.name == ".env" or file.suffix.lower() == ".env":
                    pat = LEGACY_ENV_DOTENV
                else:
                    pat = LEGACY_ENV_CODE
            for i, line in enumerate(lines, start=1):
                if pat.search(line):
                    rel = file.relative_to(ROOT).as_posix()
                    hits[key].append((rel, i, line.strip()[:300]))
    return hits


if __name__ == "__main__":
    try:
        # Windows terminals may default to cp1252 which can't print emojis.
        # Reconfigure stdout to avoid UnicodeEncodeError during printing.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
        scan_all = "--all" in sys.argv
        hits = scan(scan_all=scan_all)
        if not scan_all:
            print("(fast mode) Scanning current pipeline surfaces only. Use --all to include legacy trees.")
        for key, rows in hits.items():
            print(f"\n== {key} ({len(rows)} hits) ==")
            for rel, i, line in rows[:200]:
                print(f"{rel}:{i}: {line}")
            if len(rows) > 200:
                print(f"... truncated ({len(rows) - 200} more)")
    except BrokenPipeError:
        # Allows piping output to tools like Select-Object without a noisy stacktrace.
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
