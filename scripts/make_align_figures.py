"""Figures for the 2026-11-18 alignment / reading-copy changes.

  python scripts/make_align_figures.py --out reports/metric_route_20261117

align_<sample>.jpg   11-17 defaults | 11-18 defaults, with vertical guides
cleanup_fix_IMG_1650.jpg   reading copy without | with content protection
"""
import argparse
import glob
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.core.metric_pipeline import dewarp_metric, method_from_spec  # noqa: E402
from scoredewarp.core.preprocess import clean_page, paper_region  # noqa: E402

OLD = "gcs:c.spacing_gate=0.2,c.ends_w=0.0,c.keep_scale=false,j.vseg_w=4.0,m.lam_u=3.0"  # 11-17 defaults
DEV = ["dev/0062_re_jianpu_moderate", "dev/0033_re_mixed_severe"]


def panel(im, label, height=1000, guides=True):
    h, w = im.shape[:2]
    im = cv2.resize(im, (int(w * height / h), height), interpolation=cv2.INTER_AREA)
    if guides:
        for x in np.linspace(0, im.shape[1], 13)[1:-1]:
            cv2.line(im, (int(x), 0), (int(x), height), (60, 60, 230), 1)
    bar = np.full((44, im.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, label, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2)
    return np.vstack([bar, im])


def row(panels):
    gap = np.full((panels[0].shape[0], 16, 3), 255, np.uint8)
    out = [panels[0]]
    for p in panels[1:]:
        out += [gap, p]
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    old = method_from_spec(OLD)
    for s in DEV:
        img = cv2.imread(str(root / "data/bench/doc3d_score" / s / "photo.jpg"))
        cv2.setRNGSeed(0)   # GrabCut (paper mask) is randomised
        a = old(img).image
        cv2.setRNGSeed(0)
        b = dewarp_metric(img).image
        fig = row([panel(a, "11-17 defaults"), panel(b, "11-18 defaults")])
        cv2.imwrite(str(args.out / f"align_{s.replace('/', '_')}.jpg"), fig, [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(s, flush=True)
    cpms = glob.glob(str(root / "data/external/cpms/source/CPMS-*/data"))[0]
    img = cv2.imread(glob.glob(f"{cpms}/IMG_1650.*")[0])
    cv2.setRNGSeed(0)
    res = dewarp_metric(img)
    sp = res.structures.spacing_est
    before = clean_page(res.image, paper_region(res.paper, res.image, spacing=sp, protect=False), sp)
    after = clean_page(res.image, paper_region(res.paper, res.image, spacing=sp), sp)
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)  # noqa: E731
    fig = row([panel(res.image, "rectified", guides=False), panel(g(before), "reading copy, 11-17", guides=False),
               panel(g(after), "reading copy, 11-18", guides=False)])
    cv2.imwrite(str(args.out / "cleanup_fix_IMG_1650.jpg"), fig, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("IMG_1650")


if __name__ == "__main__":
    main()
