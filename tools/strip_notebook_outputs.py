from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def strip_outputs(nb: dict[str, Any]) -> tuple[dict[str, Any], int]:
    changed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if "outputs" in cell and cell["outputs"]:
            cell["outputs"] = []
            changed += 1
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed += 1
        meta = cell.get("metadata")
        if isinstance(meta, dict) and "execution" in meta:
            meta.pop("execution", None)
            changed += 1
    return nb, changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Strip code cell outputs/execution counts from a Jupyter notebook.")
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--inplace", action="store_true", help="Modify the notebook in-place")
    ap.add_argument("--backup", action="store_true", help="Write a .bak copy before modifying")
    args = ap.parse_args()

    path = args.notebook
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)

    nb2, changed = strip_outputs(nb)

    if not args.inplace:
        print(changed)
        return 0

    if args.backup:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(raw, encoding="utf-8")

    path.write_text(json.dumps(nb2, ensure_ascii=False, indent=2), encoding="utf-8")
    print(changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
