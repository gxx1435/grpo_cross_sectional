"""
Build a CSI 500 stock-pool view of the 1-minute data.

Reads ``files/csi500_constituents.csv`` and, for every constituent that has
a CSV in ``files/2025/`` or ``files/2026/1分钟/``, places a hard-link inside

    files/csi500/2025/<filename>.csv
    files/csi500/2026_1min/<filename>.csv

Hard links don't copy bytes, so the pool view costs (almost) zero extra
disk and stays in sync with the source files. Re-run this script after the
CSI 500 list is rebalanced or after new bars arrive.
"""

import csv
import os
from pathlib import Path
from typing import List, Tuple


HERE = Path(__file__).resolve().parent
CONSTITUENTS_CSV = HERE / "csi500_constituents.csv"

SOURCES: List[Tuple[Path, Path]] = [
    (HERE / "2025",         HERE / "csi500" / "2025"),
    (HERE / "2026" / "1分钟", HERE / "csi500" / "2026_1min"),
]


def read_constituents(path: Path) -> List[str]:
    """Return the list of expected file names (e.g. ``sh600000.csv``)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["filename"] for row in reader]


def link_pool(src_dir: Path, dst_dir: Path, filenames: List[str]) -> Tuple[int, List[str]]:
    """Hard-link every present file into ``dst_dir``. Returns (n_linked, missing)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_linked = 0
    missing: List[str] = []

    for fname in filenames:
        src = src_dir / fname
        dst = dst_dir / fname

        if not src.is_file():
            missing.append(fname)
            continue

        if dst.exists() or dst.is_symlink():
            dst.unlink()

        try:
            os.link(src, dst)
        except OSError:
            os.symlink(src, dst)
        n_linked += 1

    return n_linked, missing


def main() -> None:
    if not CONSTITUENTS_CSV.is_file():
        raise SystemExit(
            f"missing {CONSTITUENTS_CSV}. "
            "Re-pull constituents first (akshare.index_stock_cons_csindex)."
        )

    filenames = read_constituents(CONSTITUENTS_CSV)
    print(f"CSI 500 constituents: {len(filenames)} stocks")
    print("-" * 70)

    for src_dir, dst_dir in SOURCES:
        if not src_dir.is_dir():
            print(f"skip   : {src_dir} (not a directory)")
            continue

        n_linked, missing = link_pool(src_dir, dst_dir, filenames)
        rel_dst = dst_dir.relative_to(HERE)
        print(f"linked : {n_linked:>4} files -> files/{rel_dst}")
        if missing:
            print(f"missing: {len(missing)} files (no source data found)")
            for fname in missing[:5]:
                print(f"         - {fname}")
            if len(missing) > 5:
                print(f"         ... and {len(missing) - 5} more")

            log_path = dst_dir.parent / f"missing_{dst_dir.name}.txt"
            log_path.write_text("\n".join(missing), encoding="utf-8")
            print(f"         full list written to files/{log_path.relative_to(HERE)}")
        print()


if __name__ == "__main__":
    main()
