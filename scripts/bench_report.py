"""Merge bench runs (rows.jsonl) and print one comparison table.

  python scripts/bench_report.py outputs/bench/dev_legacy outputs/bench/dev_gcs_v1 \
      --methods input abc abc_mesh gcs gcs_c --base abc
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--base", default=None, help="method for paired comparison (default: first non-input)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        rows += [json.loads(l) for l in (run / "rows.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["method"] in args.methods]
    methods = list(args.methods)
    if args.base:
        methods.remove(args.base)
        methods.insert(1 if methods and methods[0] == "input" else 0, args.base)
    rep = summarize(rows, methods)
    print(rep)
    if args.out:
        args.out.write_text(rep)


if __name__ == "__main__":
    main()
