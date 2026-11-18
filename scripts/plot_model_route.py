"""Figures for reports/model_route_20261119.md from a bench run with methods
metric / models_page / models_all (outputs/bench/final_models_20261119).

    python scripts/plot_model_route.py --run outputs/bench/final_models_20261119 --out reports/model_route_20261119
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = [("metric", "no models"), ("models_page", "page model"), ("models_all", "page + staff net")]


def panel(run, sample, rows, out):
    name = sample.split("/")[1]
    ims = [cv2.imread(str(ROOT / f"data/bench/doc3d_score/dev/{name}/photo.jpg"))]
    labels = ["input"]
    for m, lab in METHODS:
        ims.append(cv2.imread(str(run / "images" / "dev" / name / f"{m}.jpg")))
        r = rows[(sample, m)]
        labels.append(f"{lab}: AD {r['ad_c']:.2f}  LD {r['ld_c']:.1f}")
    ims = [cv2.resize(i, (int(420 * i.shape[1] / i.shape[0]), 420), interpolation=cv2.INTER_AREA) for i in ims]
    tiles = []
    for im, lab in zip(ims, labels):
        head = np.full((34, im.shape[1], 3), 255, np.uint8)
        cv2.putText(head, lab, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
        tiles.append(cv2.copyMakeBorder(np.vstack([head, im]), 0, 0, 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    cv2.imwrite(str(out / f"{name}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", nargs="*", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = {(r["sample"], r["method"]): r for r in map(json.loads, open(a.run / "rows.jsonl"))}
    samples = sorted({s for s, _ in rows})
    if a.samples:
        pick = [s for s in samples if s.split("/")[1][:4] in a.samples]
    else:   # largest gains and losses of the page model vs no models
        d = sorted(samples, key=lambda s: rows[(s, "models_page")]["ad_c"] - rows[(s, "metric")]["ad_c"])
        pick = d[:4] + d[-3:]
    for s in pick:
        panel(a.run, s, rows, a.out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, lab in METHODS:
        v = np.sort([rows[(s, m)]["ad_c"] for s in samples])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=lab)
    ax.set_xlabel("AD (content box, % of page width)")
    ax.set_ylabel("fraction of dev pages")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / "ad_cdf.png", dpi=130)
    print([s.split("/")[1] for s in pick])


if __name__ == "__main__":
    main()
