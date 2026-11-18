"""Four-panel figures: input | mainline before the margin fix | mainline now |
reading copy (cleanup of the mainline output).

  python scripts/make_compare_figures.py --out reports/compare_20261118c

Red guides: vertical lines through the left / right end of the top staff row
(``field.margin_stats``); on a correct page every staff starts and ends on
them. The labels give the fitted slant of both margins and the width change
from the top to the bottom staff. 6 CPMS photos (random, seed 7, excluding
those already in the route report) and 6 dev pages across kinds and tiers.
The reading copy is what scripts/dewarp.py writes as <name>_enhanced.png.
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

OLD = "gcs:stem_refine=false,c.ends_mode=chain"   # 11-18b defaults (reports/compare_20261118b)
DEV = ["0004_de_staff_severe", "0031_re_staff_severe", "0027_re_jianpu_moderate",
       "0107_re_jianpu_severe", "0013_re_mixed_severe", "0003_re_mixed_mild"]


def guides(img, S):
    """Draw the top row's margin lines; return (image, label suffix)."""
    from scoredewarp.core.field import margin_stats
    ms = margin_stats(S)
    if ms is None:
        return img, ""
    out = img.copy()
    t = max(2, img.shape[1] // 500)
    for side in ("left", "right"):
        x = int(round(ms[side][0]))
        cv2.line(out, (x, 0), (x, img.shape[0] - 1), (0, 0, 255), t)
    return out, f"  L{ms['left'][2]:+.1f} R{ms['right'][2]:+.1f} deg, w{ms['keystone'] * 100:+.1f}%"


def job(arg):
    name, path, out = arg
    from make_route_figures import tile
    from scoredewarp.core.metric_pipeline import dewarp_metric, method_from_spec
    from scoredewarp.core.preprocess import clean_page, paper_region
    img = cv2.imread(str(path))
    cv2.setRNGSeed(0)
    old = method_from_spec(OLD)(img)
    cv2.setRNGSeed(0)
    res = dewarp_metric(img)
    sp = getattr(res.structures, "spacing_est", None)
    clean = clean_page(res.image, paper_region(res.paper, res.image, spacing=sp, content=res.content), sp)
    a, la = guides(old.image, old.structures)
    b, lb = guides(res.image, res.structures)
    fig = tile([img, a, b, cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)],
               ["input", "before" + la, "now" + lb, "reading copy (now)"])
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
