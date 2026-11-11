"""Dewarp score photos.

    python scripts/dewarp.py photo1.jpg [photo2.jpg ...] --out outputs/run
Writes per input:
    <name>_rectified.png   rectified colour image
    <name>_enhanced.png    illumination-normalised grayscale (for OMR / reading)
    <name>_debug.jpg       detected structures on the rectified image
    <name>_result.json     page type, regions, staff lines, jianpu rows, timings
    <name>_bmap.npy        backward map (output grid stride 4 -> input pixels)
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoredewarp.core.pipeline import Config, dewarp, refine_dewarp, result_json  # noqa: E402
from scoredewarp.core.structures import draw_structures  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--no-quad", action="store_true", help="skip page-boundary homography")
    ap.add_argument("--no-surface", action="store_true")
    ap.add_argument("--no-field", action="store_true")
    ap.add_argument("--rot90", action="store_true", help="also test 90-degree page rotation")
    ap.add_argument("--checkpoint", help="learned initialization followed by C refinement; skips A/B")
    ap.add_argument("--network-only", action="store_true", help="with --checkpoint, skip C refinement")
    args = ap.parse_args()
    if args.network_only and not args.checkpoint:
        ap.error("--network-only requires --checkpoint")
    os.makedirs(args.out, exist_ok=True)
    cfg = Config(use_page_quad=not args.no_quad, use_surface=not args.no_surface, use_field=not args.no_field,
                 test_rot90=args.rot90)
    if args.checkpoint:
        from scoredewarp.learn.infer import load, predict
        net, size = load(args.checkpoint)
        cfg.use_field = not (args.no_field or args.network_only)
    for path in args.inputs:
        img = cv2.imread(path)
        if img is None:
            print("cannot read", path)
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        if os.path.basename(path) == "photo.jpg":  # synthetic folders
            name = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if args.checkpoint:
            start = time.perf_counter()
            bm, _, _ = predict(net, img, size, out_size=(1600, 2263))
            network_time = time.perf_counter() - start
            res = refine_dewarp(img, bm, cfg)
            res.timings["network_init"] = network_time
            res.timings["total"] += network_time
            res.stages["checkpoint"] = os.path.abspath(args.checkpoint)
        else:
            res = dewarp(img, cfg)
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
