"""Merge evaluation CSVs (e.g. shards run on several Kaggle kernels) into one Markdown table.

    uv run python scripts/summarize.py --name speed outputs/kaggle/*/eval_speed_*.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluate import summarize

from robothand_sim.paths import OUTPUT_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("csvs", nargs="+", type=Path)
    args = ap.parse_args()

    rows = [r for p in args.csvs for r in csv.DictReader(p.open())]
    seeds = sorted({int(r["seed"]) for r in rows})
    beans = max(int(r["index"]) for r in rows) + 1
    md = summarize(rows, f"eval: {args.name}", f"layout seeds {seeds[0]}-{seeds[-1]} x {beans} beans per condition")
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / f"eval_{args.name}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
