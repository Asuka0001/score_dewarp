"""Demo images for the README / GitHub: original, rectified, cleanup and A4 page
per photo, plus one side-by-side panel each.

Run dewarp first, then point this script at the same photos and output folder:

  python scripts/dewarp.py photos/*.HEIC --out outputs/demo --models page --crop-content
  python scripts/make_demo.py photos/*.HEIC --results outputs/demo --dest demo

Images are re-encoded as JPEG (long side ``--size``), which also drops the
photos' EXIF metadata (GPS position, device).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoredewarp.core.image_input import read_photo  # noqa: E402

VIEWS = (("original", "Original"), ("rectified", "Rectified"), ("enhanced", "Cleanup"), ("page", "A4 page"))


def shrink(im, long_side):
    s = long_side / max(im.shape[:2])
    return im if s >= 1 else cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def panel(ims, h):
    cols = []
    for (_, label), im in zip(VIEWS, ims):
        im = cv2.resize(im, (int(round(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_AREA)
        bar = np.full((48, im.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        cols += [np.vstack([bar, im]), np.full((h + 48, 12, 3), 210, np.uint8)]
    return np.hstack(cols[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--results", required=True, help="dewarp.py --out folder")
    ap.add_argument("--dest", default="demo")
    ap.add_argument("--size", type=int, default=1600, help="long side of each exported image (px)")
    ap.add_argument("--panel-height", type=int, default=900)
    args = ap.parse_args()
    dest = Path(args.dest)
    (dest / "images").mkdir(parents=True, exist_ok=True)
    q = [cv2.IMWRITE_JPEG_QUALITY, 88]
    for path in args.photos:
        name = Path(path).stem
        ims = [read_photo(path)]
        for key, _ in VIEWS[1:]:
            im = cv2.imread(os.path.join(args.results, f"{name}_{key}.png"))
            if im is None:
                sys.exit(f"missing {name}_{key}.png in {args.results}")
            ims.append(im)
        for (key, _), im in zip(VIEWS, ims):
            cv2.imwrite(str(dest / "images" / f"{name}_{key}.jpg"), shrink(im, args.size), q)
        cv2.imwrite(str(dest / f"{name}_panel.jpg"), panel(ims, args.panel_height), q)
        print(name)


if __name__ == "__main__":
    main()
