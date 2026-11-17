"""Four-panel figures: input | legacy ABC | mainline 11-17 defaults | mainline now.

  python scripts/make_compare_figures.py --out reports/compare_20261118

6 CPMS photos (random, seed 7, excluding those already in the route report)
and 6 dev pages across kinds and difficulty tiers.
"""
import argparse
import glob
from multiprocessing import Pool
from pathlib import Path
import random
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

OLD = "gcs:c.spacing_gate=0.2,c.ends_w=0.0,c.keep_scale=false,c.anchor=false,j.vseg_w=4.0,m.lam_u=3.0"  # 11-17
DEV = ["0004_de_staff_severe", "0031_re_staff_severe", "0027_re_jianpu_moderate",
       "0107_re_jianpu_severe", "0013_re_mixed_severe", "0003_re_mixed_mild"]


def job(arg):
    name, path, out = arg
    from make_route_figures import tile
    from scoredewarp.core.metric_pipeline import dewarp_metric, method_from_spec
    from scoredewarp.core.pipeline import dewarp
    img = cv2.imread(str(path))
    legacy = dewarp(img).image
    cv2.setRNGSeed(0)
    old = method_from_spec(OLD)(img).image
    cv2.setRNGSeed(0)
    new = dewarp_metric(img).image
    fig = tile([img, legacy, old, new], ["input", "legacy ABC", "mainline 11-17", "mainline 11-18 (now)"])
    cv2.imwrite(str(Path(out) / f"{name}.jpg"), fig, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cpms = sorted(glob.glob(str(ROOT / "data/external/cpms/source/CPMS-*/data/*")))
    cpms = [p for p in cpms if Path(p).stem not in ("IMG_1650", "IMG_1615", "IMG_1709")]
    random.Random(7).shuffle(cpms)
    jobs = [(f"cpms_{Path(p).stem}", p, args.out) for p in cpms[:6]]
    jobs += [(f"dev_{s}", ROOT / "data/bench/doc3d_score/dev" / s / "photo.jpg", args.out) for s in DEV]
    with Pool(args.workers) as pool:
        for name in pool.imap_unordered(job, jobs):
            print(name, flush=True)


if __name__ == "__main__":
    main()
