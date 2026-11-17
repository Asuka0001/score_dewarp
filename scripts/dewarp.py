"""Dewarp score photos.

    python scripts/dewarp.py photo1.jpg [photo2.jpg ...] --out outputs/run
Writes per input:
    <name>_rectified.png   rectified colour image
    <name>_enhanced.png    illumination-normalised grayscale (for OMR / reading)
    <name>_debug.jpg       detected structures on the rectified image
    <name>_result.json     page type, regions, staff lines, jianpu rows, timings
    <name>_bmap.npy        backward map (output grid stride 4 -> input pixels)

Methods
    metric       (default) page homography -> camera + developable page surface
                 fitted to staff lines / jianpu rows / vertical strokes -> residual field
    traditional  legacy A+B+C (polynomial surface), kept for comparison only
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoredewarp.core.metric_pipeline import MetricConfig, dewarp_metric  # noqa: E402
from scoredewarp.core.pipeline import Config, dewarp, result_json  # noqa: E402
from scoredewarp.core.structures import draw_structures  # noqa: E402
from scoredewarp.core.transforms import mapping_quality  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--method", choices=("metric", "traditional"), default="metric")
    ap.add_argument("--no-field", action="store_true", help="skip the residual field after the surface fit")
    ap.add_argument("--crop-content", action="store_true",
                    help="metric: output only the detected content box instead of the whole page")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for path in args.inputs:
        img = cv2.imread(path)
        if img is None:
            print("cannot read", path)
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        if os.path.basename(path) == "photo.jpg":  # benchmark / synthetic folders
            name = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if args.method == "metric":
            cfg = MetricConfig(residual="none" if args.no_field else MetricConfig.residual,
                               full_page=not args.crop_content)
            res = dewarp_metric(img, cfg)
        else:
            res = dewarp(img, Config(use_field=not args.no_field))
        res.stages["method"] = args.method
        res.stages["mapping_quality"] = mapping_quality(res.bmap, (img.shape[1], img.shape[0]))
        base = os.path.join(args.out, name)
        cv2.imwrite(base + "_rectified.png", res.image)
        cv2.imwrite(base + "_enhanced.png", res.enhanced)
        cv2.imwrite(base + "_debug.jpg", draw_structures(res.image, res.structures))
        np.save(base + "_bmap.npy", res.bmap.grid.astype(np.float32))
        with open(base + "_result.json", "w") as f:
            json.dump(result_json(res), f, ensure_ascii=False, indent=1, default=float)
        print(f"{path}: type={res.page_type} size={res.bmap.out_size} total={res.timings['total']:.1f}s")


if __name__ == "__main__":
    main()
