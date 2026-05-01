import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "databricks.yml").exists() or (parent / ".git").exists():
            return parent
    return cur


def resolve_path(env_value: str | None, default: Path, root: Path) -> Path:
    if env_value is None or env_value.strip() == "":
        return default
    p = Path(env_value)
    return p if p.is_absolute() else (root / p)


def run_step(script_path: Path, repo_root: Path) -> None:
    print("=" * 88)
    print(f"RUN: {script_path.relative_to(repo_root)}")
    print("=" * 88)
    subprocess.run([sys.executable, str(script_path)], cwd=str(repo_root), check=True)


def main() -> int:
    load_dotenv()

    root = find_repo_root(Path(__file__).resolve())

    scripts_dir = root / "scripts"
    out_dir = resolve_path(
        os.getenv("PAPERMILL_OUTPUT_DIR"), root / "data" / "gold" / "papermill_outputs", root
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Canonical .py step scripts (preferred for deployment).
    steps = [
        scripts_dir / "bronze" / "crawl_dwd.py",
        scripts_dir / "bronze" / "crawl_lubw.py",
        scripts_dir / "silver" / "clean_dwd_daten.py",
        scripts_dir / "silver" / "create_messungen_complete.py",
        scripts_dir / "gold" / "data_full.py",
        scripts_dir / "gold" / "create_masterdata.py",
    ]

    missing = [p for p in steps if not p.exists()]
    if missing:
        joined = "\n".join(f"- {p.relative_to(root)}" for p in missing)
        raise FileNotFoundError(
            "Missing .py pipeline step scripts under scripts/.\n\n" f"Missing:\n{joined}"
        )

    print(f"Using repo root: {root}")
    print(f"Outputs dir (kept for compatibility): {out_dir}")

    for step in steps:
        run_step(step, root)

    print("=" * 88)
    print("PIPELINE OK (.py only)")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
