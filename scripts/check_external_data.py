"""Validate prepared splits, coordinate ranges, labels and source-group separation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import cv2
from scipy.interpolate import LinearNDInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.learn.dataset import build_targets


def preview(root, manifest):
    """Show flat/photo/GT inversion to make coordinate-direction errors visible."""
    rows = []
    for split in ("train", "val", "test"):
        row = next(x for x in manifest["samples"] if x["path"].startswith(split + "/"))
        folder = root / row["path"]
        photo = cv2.imread(str(folder / "photo.jpg"))
        mapping = np.load(folder / "gt_bmap_s2.npy").astype(np.float32)[::2, ::2]
        h, w = mapping.shape[:2]
        valid = np.isfinite(mapping).all(axis=-1)
        yy, xx = np.mgrid[:h, :w]
        # The stored map samples the whole input photo on a regular grid.
        source = np.stack([xx * photo.shape[1] / w, yy * photo.shape[0] / h], axis=-1)
        inverse = LinearNDInterpolator(mapping[valid], source[valid], fill_value=-1000)
        wf, hf = row["size"]
        gx, gy = np.meshgrid(np.linspace(0, wf, 360), np.linspace(0, hf, 480))
        coords = inverse(gx, gy).astype(np.float32)
        rectified = cv2.remap(photo, coords[..., 0], coords[..., 1], cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        flat_path = folder / "flat.png"
        if flat_path.exists():
            flat = cv2.imread(str(flat_path))
            first_title = "flat reference"
        else:
            # Doc3D img+uv does not include the original flat texture. Show UV
            # coordinates instead of an empty tile that looks like missing data.
            flat = np.zeros((h, w, 3), np.uint8)
            flat[valid, 2] = np.clip(mapping[valid, 0] / wf * 255, 0, 255).astype(np.uint8)
            flat[valid, 1] = np.clip(mapping[valid, 1] / hf * 255, 0, 255).astype(np.uint8)
            first_title = "flat coordinates (UV)"
        tiles = []
        for title, image in ((first_title, flat), ("simulated photo" if flat_path.exists() else "photo", photo),
                             ("GT rectified", rectified)):
            tile = cv2.resize(image, (360, 480), interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, f"{split}: {title}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 1)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(root / "preview.jpg"), np.concatenate(rows, axis=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("--preview", action="store_true", help="also render flat/photo/GT inversion contact sheet")
    args = ap.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    groups, counts = {}, {s: 0 for s in ("train", "val", "test")}
    coverage = []
    for row in manifest["samples"]:
        split = row["path"].split("/")[0]
        group = row["source_group"]
        if group in groups and groups[group] != split:
            raise ValueError(f"Source group leaks across splits: {group}")
        groups[group] = split
        d = build_targets(str(args.data / row["path"]))
        if not all(np.isfinite(x).all() for x in d.values()):
            raise ValueError(f"Nonfinite tensor: {row['path']}")
        page = d["page"][0] > 0
        values = d["fwd"][:, page]
        if page.mean() < 0.01 or values.min() < 0 or values.max() > 1:
            raise ValueError(f"Invalid coordinate map: {row['path']}")
        for head, valid in row["supervision"].items():
            if bool(d[head + "_valid"]) != valid:
                raise ValueError(f"Invalid supervision flag: {row['path']}")
        counts[split] += 1
        coverage.append(float(page.mean()))
    if counts != manifest["counts"]:
        raise ValueError("Manifest sample count mismatch")
    actual = {str(p.parent.relative_to(args.data)) for p in args.data.glob("*/*/gt.json")}
    if actual != {r["path"] for r in manifest["samples"]}:
        raise ValueError("Unlisted or missing prepared samples")
    report = {"counts": counts, "source_groups": len(groups), "group_overlap": 0,
              "page_coverage_min": min(coverage), "page_coverage_mean": float(np.mean(coverage)),
              "finite_and_in_range": True}
    (args.data / "validation.json").write_text(json.dumps(report, indent=2))
    if args.preview:
        preview(args.data, manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
