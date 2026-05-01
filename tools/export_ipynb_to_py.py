import argparse
import json
import os
from pathlib import Path


DEFAULT_NOTEBOOKS = [
    Path("scripts/bronze/crawl_dwd.ipynb"),
    Path("scripts/bronze/crawl_lubw.ipynb"),
    Path("scripts/silver/clean_dwd_daten.ipynb"),
    Path("scripts/silver/create_messungen_complete.ipynb"),
    Path("scripts/gold/data_full.ipynb"),
    Path("scripts/gold/create_masterdata.ipynb"),
]


AUTO_GENERATED_MARKER = "# Auto-generated from:"


def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "databricks.yml").exists() or (parent / ".git").exists():
            return parent
    return cur


def _cell_source_as_str(cell_source) -> str:
    if isinstance(cell_source, list):
        return "".join(cell_source)
    if isinstance(cell_source, str):
        return cell_source
    return ""


_STRIP_PREFIXES = ("%", "!")


def _strip_magics(code: str) -> tuple[str, int]:
    """Remove IPython magics/shell escapes that would break plain Python."""
    stripped = []
    removed = 0
    for line in code.splitlines(keepends=True):
        if line.lstrip().startswith(_STRIP_PREFIXES):
            removed += 1
            stripped.append(f"# [stripped notebook magic] {line}")
        else:
            stripped.append(line)
    return "".join(stripped), removed


def export_notebook_to_py(in_path: Path, out_path: Path) -> None:
    nb = json.loads(in_path.read_text(encoding="utf-8"))

    cells = nb.get("cells", [])
    code_cells: list[str] = []
    total_magics_removed = 0

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = _cell_source_as_str(cell.get("source", ""))
        src, removed = _strip_magics(src)
        total_magics_removed += removed
        if src.strip() == "":
            continue
        code_cells.append(src.rstrip() + "\n")

    combined = "\n\n".join(code_cells).rstrip() + "\n"

    # Move any `from __future__ import ...` statements to the top of the file.
    future_imports: list[str] = []
    rest_lines: list[str] = []
    for line in combined.splitlines(keepends=True):
        if line.lstrip().startswith("from __future__ import "):
            future_imports.append(line)
        else:
            rest_lines.append(line)
    rest = "".join(rest_lines)

    header = (
        f"{AUTO_GENERATED_MARKER} "
        + in_path.as_posix()
        + os.linesep
        + "# NOTE: This file is a notebook export snapshot. Prefer editing the canonical .py pipeline code."
        + os.linesep
    )
    if total_magics_removed:
        header += f"# Stripped {total_magics_removed} notebook magic lines (%/!).\n"
    header += "\n"

    # Always provide a display() fallback; many notebooks use it.
    prelude = """\
try:\n    from IPython.display import display  # type: ignore\nexcept Exception:  # pragma: no cover\n    def display(x=None):\n        print(x)\n\n"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + "".join(future_imports) + "\n" + prelude + rest, encoding="utf-8")


def _is_auto_generated_py(py_path: Path) -> bool:
    try:
        head = py_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]
        return any(AUTO_GENERATED_MARKER in line for line in head)
    except Exception:
        return False


def notebook_paths(repo_root: Path, all_notebooks: bool) -> list[Path]:
    if not all_notebooks:
        return [repo_root / p for p in DEFAULT_NOTEBOOKS]

    notebooks: list[Path] = []
    for p in (repo_root / "scripts").rglob("*.ipynb"):
        # skip deprecated/legacy areas by default
        if any(part in {"deprecated", "papermill_outputs"} for part in p.parts):
            continue
        notebooks.append(p)
    return sorted(notebooks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export selected pipeline notebooks (code cells only) to .py scripts"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all scripts/**/*.ipynb (excluding deprecated/papermill_outputs)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .py files",
    )
    parser.add_argument(
        "--overwrite-any",
        action="store_true",
        help=(
            "Allow overwriting existing .py files even if they are NOT marked as auto-generated. "
            "Use with care; this can clobber canonical pipeline scripts."
        ),
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    nbs = notebook_paths(repo_root, all_notebooks=args.all)

    exported = 0
    skipped = 0
    protected = 0
    for nb_path in nbs:
        if not nb_path.exists():
            continue
        py_path = nb_path.with_suffix(".py")
        if py_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if (
            py_path.exists()
            and args.overwrite
            and not args.overwrite_any
            and not _is_auto_generated_py(py_path)
        ):
            protected += 1
            continue
        export_notebook_to_py(nb_path, py_path)
        exported += 1

    print(f"Repo root: {repo_root}")
    print(f"Exported: {exported}")
    print(f"Skipped (exists, use --overwrite): {skipped}")
    if protected:
        print(
            "Protected (refused to overwrite non-auto-generated .py; use --overwrite-any): "
            f"{protected}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
