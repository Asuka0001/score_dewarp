"""Generate a synthetic benchmark: flat page + simulated photo + GT annotations.

    python scripts/make_synth.py --out data/synth --n 6
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoredewarp.synth.distort import simulate_photo  # noqa: E402
from scoredewarp.synth.render import render_page  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n", type=int, default=6, help="pages per kind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for kind in ["staff", "jianpu", "mixed"]:
        for i in range(args.n):
            name = f"{kind}_{i:03d}"
            flat, ann, meta = render_page(kind, rng)
            sim = simulate_photo(flat, ann, rng, strength=args.strength)
            d = os.path.join(args.out, name)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "flat.png"), flat)
            cv2.imwrite(os.path.join(d, "photo.jpg"), sim["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            np.save(os.path.join(d, "gt_bmap_s2.npy"), sim["bmap_lo"].astype(np.float16))
            with open(os.path.join(d, "gt.json"), "w") as f:
                json.dump({"meta": meta, "flat_ann": ann, "photo_ann": sim["ann"]}, f)
            print("wrote", d)


if __name__ == "__main__":
    main()
