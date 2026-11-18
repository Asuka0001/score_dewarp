"""Reproducible detection/geometry ablations on dev, with input-only map checks.

No thresholds are selected from ground truth. The replayed guard rejects a
learned output only when its local map check fails and the classical output
passes. It is a diagnostic policy, not a default pipeline change.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench


SPECS = {
    "classical": "metric",
    "homr": "gcs:seg=homr,seg_parts=staff+stem",
    "stem_disconnected": "gcs:seg=homr,seg_parts=stem,stem_bridge=false",
    "stem_connected": "gcs:seg=homr,seg_parts=stem,stem_bridge=true",
    "homr_rows": "gcs:seg=homr,seg_parts=staff+stem,gcs_ends=rows",
    "homr_noends": "gcs:seg=homr,seg_parts=staff+stem,w.ends=0",
}
# Fixed before evaluating new results. These are diagnostic gates, not
# calibrated estimates of geometric accuracy or global injectivity.
GUARD = {"max_fold_fraction": 0.0, "max_anisotropy_p99": 20.0}
PICTURES = {"0004", "0021", "0041", "0098", "0031", "0013", "0027", "0107"}


def map_pass(row):
    q = row.get("notes", {}).get("mapping_quality", {})
    return ("error" not in row and q.get("checked_cells", 0) > 0
            and q.get("folded_cell_fraction", 1) <= GUARD["max_fold_fraction"]
            and q.get("anisotropy_p99", float("inf")) <= GUARD["max_anisotropy_p99"])


def worker(folder, names, out):
    bench.register_new_methods()
    folder = Path(folder)
    save = str(Path(out) / "images") if folder.name[:4] in PICTURES else None
    rows = bench.run_one((str(folder), [SPECS[n] for n in names], save, True))
    for name, row in zip(names, rows):
        row["spec"] = row["method"]
        row["method"] = name
    return rows


def stats(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return [float("nan")] * 4
    return [float(x) for x in (np.median(a), a.mean(), np.percentile(a, 90), a.max())]


def guard_choice(base, learned):
    bad = not map_pass(learned)
    use_base = bad and map_pass(base)
    return ("classical" if use_base else "homr"), bool(bad and not use_base)


def report(rows, names):
    by = {(r["sample"], r["method"]): r for r in rows}
    samples = sorted({r["sample"] for r in rows})
    selected = []
    fallback, unresolved = [], []
    if {"classical", "homr"} <= set(names):
        for s in samples:
            base, learned = by[s, "classical"], by[s, "homr"]
            choice, still_bad = guard_choice(base, learned)
            use_base = choice == "classical"
            chosen = dict(base if use_base else learned)
            chosen["method"] = "homr_map_guard"
            chosen["chosen_method"] = "classical" if use_base else "homr"
            selected.append(chosen)
            if use_base:
                fallback.append(s)
            elif still_bad:
                unresolved.append(s)
        rows = rows + selected
        names = names + ["homr_map_guard"]
    lines = ["# Detection / geometry experiments", "",
             "Fresh runs on dev; canvas resolution. Lower AD/LD is better.", "",
             "| method | n | errors | AD median | mean | p90 | max | LD median | mean | p90 | max |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        group = [r for r in rows if r["method"] == name]
        vals = [stats([r[k] for r in group if r.get(k) is not None]) for k in ("ad_c", "ld_c")]
        lines.append(f"| {name} | {len(group)} | {sum('error' in r for r in group)} | "
                     + " | ".join(f"{v:.3f}" for part in vals for v in part) + " |")
    comparisons = [(n, "classical") for n in names if n != "classical"] if "classical" in names else []
    if {"stem_disconnected", "stem_connected"} <= set(names):
        comparisons.append(("stem_connected", "stem_disconnected"))
    if {"homr", "homr_rows"} <= set(names):
        comparisons.append(("homr_rows", "homr"))
    if {"homr", "homr_noends"} <= set(names):
        comparisons.append(("homr_noends", "homr"))
    by.update({(r["sample"], r["method"]): r for r in selected})
    lines += ["", "| paired AD comparison | n | better | same | worse | better >0.3 | worse >0.3 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, base in comparisons:
        delta = np.array([by[s, name]["ad_c"] - by[s, base]["ad_c"] for s in samples
                          if by[s, name].get("ad_c") is not None and by[s, base].get("ad_c") is not None])
        lines.append(f"| {name} vs {base} | {len(delta)} | {(delta < -1e-6).sum()} | "
                     f"{(abs(delta) <= 1e-6).sum()} | {(delta > 1e-6).sum()} | "
                     f"{(delta < -.3).sum()} | {(delta > .3).sum()} |")
    lines += ["", "| kind | method | n | AD median | AD mean | LD median | page cover median |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for kind in sorted({r["kind"] for r in rows}):
        for name in names:
            group = [r for r in rows if r["kind"] == kind and r["method"] == name]
            cells = []
            for k, index in (("ad_c", 0), ("ad_c", 1), ("ld_c", 0), ("cover", 0)):
                cells.append(stats([r[k] for r in group if r.get(k) is not None])[index])
            lines.append(f"| {kind} | {name} | {len(group)} | "
                         + " | ".join(f"{v:.3f}" for v in cells) + " |")
    lines += ["", f"Guard thresholds fixed before this run: `{json.dumps(GUARD)}`.",
              f"Fallbacks ({len(fallback)}): {', '.join(fallback) or 'none'}.",
              f"Both candidates fail / unresolved ({len(unresolved)}): {', '.join(unresolved) or 'none'}.",
              "Local map checks do not establish global injectivity, content preservation or geometric accuracy."]
    return "\n".join(lines) + "\n", selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/bench/doc3d_score/dev"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter", default="")
    ap.add_argument("--methods", nargs="+", choices=SPECS, default=list(SPECS))
    args = ap.parse_args()
    if args.data.name == "test":
        ap.error("this development experiment must not use the held-out test split")
    folders = sorted(p.parent for p in args.data.rglob("gt.json") if args.filter in str(p.parent))
    folders = folders[:args.limit or None]
    if not folders:
        ap.error("no samples found")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "rows.jsonl").exists():
        ap.error("output already contains rows; choose a new directory")
    diff = subprocess.check_output(["git", "diff", "--", "scoredewarp", "scripts", "tests"])
    manifest = {"data": str(args.data), "samples": [str(f) for f in folders],
                "methods": {n: SPECS[n] for n in args.methods}, "guard": GUARD,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "workers": args.workers, "seg_cache": os.environ.get("SCOREDEWARP_SEG_CACHE"),
                "seg_coreml": os.environ.get("SCOREDEWARP_SEG_COREML", "1")}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows, start = [], time.time()
    with (args.out / "rows.jsonl").open("w") as log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, str(f), args.methods, str(args.out)) for f in folders]
            for i, future in enumerate(as_completed(futures), 1):
                chunk = future.result()
                rows.extend(chunk)
                for r in chunk:
                    log.write(json.dumps(r) + "\n")
                log.flush()
                print(f"{i}/{len(folders)} samples, {time.time()-start:.0f}s, errors="
                      f"{sum('error' in r for r in rows)}", flush=True)
    summary, selected = report(rows, args.methods)
    (args.out / "summary.md").write_text(summary)
    (args.out / "guard_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    print(summary)


if __name__ == "__main__":
    main()
