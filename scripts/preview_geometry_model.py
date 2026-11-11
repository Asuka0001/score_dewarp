"""Contact sheet: input photo / ground-truth rectification / learned rectification.

Uses the first validation samples in lexical order; model rectification uses
only predicted coordinates and its predicted page mask, never GT page pixels.
"""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.eval.geometry import sample_geometry
from scoredewarp.learn.infer import load


def invert_image(image, destination, source, valid, aspect):
    if valid.sum() < 6:
        return np.full((400, 320, 3), 255, np.uint8), "not enough predicted page pixels"
    try:
        inverse = LinearNDInterpolator(destination[valid], source[valid], fill_value=-1e5)
        gx, gy = np.meshgrid(np.linspace(0, 1, 320), np.linspace(0, aspect, 400))
        coord = inverse(gx, gy).astype(np.float32)
        return cv2.remap(image, coord[..., 0], coord[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)), None
    except (ValueError, QhullError) as exc:
        return np.full((400, 320, 3), 255, np.uint8), type(exc).__name__


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    net, size = load(args.checkpoint)
    rows = []
    for folder in sorted(p for p in args.data.iterdir() if (p / "gt.json").exists())[:args.n]:
        image, meta, points, target, valid = sample_geometry(folder)
        aspect = meta["size"][1] / meta["size"][0]
        reference, _ = invert_image(image, target, points, valid, aspect)
        x = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)[..., ::-1].copy()
        with torch.no_grad():
            out = net(torch.from_numpy(x.transpose(2, 0, 1)).float()[None] / 255)
        field = out["fwd"][0].numpy().transpose(1, 2, 0) * [1, aspect]
        page = out["page"][0, 0].sigmoid().numpy() > .5
        h, w = page.shape
        yy, xx = np.mgrid[:h, :w]
        source = np.stack([xx * image.shape[1] / w, yy * image.shape[0] / h], -1)
        rectified, error = invert_image(image, field, source, page, aspect)
        tiles = []
        for title, tile in (("input", image), ("GT rectified", reference), ("model rectified", rectified)):
            tile = cv2.resize(tile, (320, 400), interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
            cv2.putText(tile, folder.name, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            if error and title == "model rectified":
                cv2.putText(tile, error, (6, 80), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, axis=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.concatenate(rows, axis=0))
    print(args.out)


if __name__ == "__main__":
    main()
