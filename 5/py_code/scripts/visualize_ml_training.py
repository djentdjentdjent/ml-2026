#!/usr/bin/env python3
"""Построить графики из готового ML run directory; обучение не повторяется."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag_nav.ml.visualize import visualize_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Путь к каталогу эксперимента")
    args = parser.parse_args()
    written = visualize_run(args.run_dir)
    if not written:
        raise SystemExit(f"В {args.run_dir} нет CSV-артефактов для визуализации")
    for path in written:
        print(path.resolve())


if __name__ == "__main__":
    main()
