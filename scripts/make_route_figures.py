"""Side-by-side figures (input | legacy ABC | metric pipeline) for the route report.

  python scripts/make_route_figures.py --out reports/metric_route_20261117
"""
import argparse
import glob
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.core.metric_pipeline import dewarp_metric  # noqa: E402
from scoredewarp.core.pipeline import dewarp  # noqa: E402

DEV = ["dev/0012_re_jianpu_moderate", "dev/0000_de_staff_moderate", "dev/0006_re_staff_severe"]
CPMS = ["IMG_1709", "IMG_1615", "IMG_1650"]


def tile(images, labels, height=900):
    out = []
    for im, lab in zip(images, labels):
        h, w = im.shape[:2]
        im = cv2.resize(im, (int(w * height / h), height), interpolation=cv2.INTER_AREA)
        bar = np.full((40, im.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, lab, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        out.append(np.vstack([bar, im]))
        out.append(np.full((height + 40, 12, 3), 255, np.uint8))
    return np.hstack(out[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    cpms = glob.glob(str(root / "data/external/cpms/source/CPMS-*/data"))[0]
    jobs = [(s.replace("/", "_"), root / "data/bench/doc3d_score" / s / "photo.jpg") for s in DEV]
    jobs += [(n, Path(glob.glob(f"{cpms}/{n}.*")[0])) for n in CPMS]
    for name, path in jobs:
        img = cv2.imread(str(path))
        legacy, new = dewarp(img), dewarp_metric(img)
        fig = tile([img, legacy.image, new.image], ["input", "legacy ABC", "metric (new)"])
        cv2.imwrite(str(args.out / f"{name}.jpg"), fig, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(name, flush=True)


if __name__ == "__main__":
    main()
