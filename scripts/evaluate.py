"""Benchmark methods/ablations on a synthetic set made by make_synth.py.

    python scripts/evaluate.py --data data/synth --out outputs/eval
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoredewarp.core.pipeline import Config, dewarp  # noqa: E402
from scoredewarp.eval.metrics import evaluate  # noqa: E402

METHODS = {
    "coarse(quad+deskew)": Config(use_surface=False, use_field=False),
    "surface3d": Config(use_field=False),
    "full(staff+jp priors)": Config(),
    "full(staff prior only)": Config(surface_priors=("staff", "vertical"), field_priors=("staff", "vertical")),
}

ORDER = ["coverage", "ad_pct", "staff_straight", "staff_spacing_cv", "barline_tilt_deg", "jp_baseline",
         "jp_height_cv", "jp_tilt_deg", "jp_dot_err", "jp_lyric_err", "jp_rowgap_cv", "type_acc", "time_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--save_images", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    samples = sorted(d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d)))
    methods = METHODS if args.methods == "all" else {k: v for k, v in METHODS.items() if k in args.methods.split(",")}
    rows = []
    for s in samples:
        d = os.path.join(args.data, s)
        gt = json.load(open(os.path.join(d, "gt.json")))
        img = cv2.imread(os.path.join(d, "photo.jpg"))
        kind = gt["meta"]["kind"]
        m = evaluate(gt, lambda p: p)
        m.update(sample=s, kind=kind, method="input(photo)")
        rows.append(m)
        for name, cfg in methods.items():
            t = time.time()
            try:
                res = dewarp(img, cfg)
            except Exception as e:  # keep benchmarking robustly
                print("FAILED", s, name, e)
                continue
            m = evaluate(gt, res.forward_points)
            m.update(sample=s, kind=kind, method=name, time_s=time.time() - t,
                     type_acc=float(res.page_type == kind))
            rows.append(m)
            if args.save_images and name.startswith("full(staff+jp"):
                cv2.imwrite(os.path.join(args.out, f"{s}_out.jpg"), res.image)
        print("done", s)
    json.dump(rows, open(os.path.join(args.out, "per_sample.json"), "w"), indent=1)
    # aggregate
    lines = []
    for kind in ["staff", "jianpu", "mixed"]:
        lines.append(f"\n### {kind}\n")
        names = ["input(photo)"] + list(methods)
        cols = [c for c in ORDER if any(c in r for r in rows if r["kind"] == kind)]
        lines.append("| method | " + " | ".join(cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for n in names:
            rs = [r for r in rows if r["kind"] == kind and r["method"] == n]
            vals = []
            for c in cols:
                v = [r[c] for r in rs if c in r and np.isfinite(r[c])]
                vals.append(f"{np.mean(v):.3f}" if v else "-")
            lines.append(f"| {n} | " + " | ".join(vals) + " |")
    txt = "\n".join(lines)
    open(os.path.join(args.out, "summary.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
