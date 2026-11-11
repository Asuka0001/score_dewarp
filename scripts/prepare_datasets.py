"""Convert external data into the existing photo.jpg/gt.json/gt_bmap_s2.npy format.

DeepScores pages are warped with our simulator; Doc3D UV is already photo->flat.
Absent score layout/structure labels are explicitly excluded from training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

# Must be set before importing OpenCV to read Doc3D EXR files.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.synth.distort import simulate_photo


def split_for(group, seed):
    bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else ("val" if bucket < 90 else "test")


def doc3d_group(stem):
    """Keep child meshes with their parent pair (official dataset issue #8).

    Names: parentA_parentB-texture-random or child_parentA_parentB-texture-random.
    """
    mesh = stem.split("-", 1)[0]
    parts = mesh.split("_")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        raise ValueError(f"Unrecognized Doc3D mesh ID: {mesh}")
    return "_".join(parts[-2:])


def doc3d_uv_to_map(uv):
    """OpenCV BGR = validity,V,U; convert Blender V to top-down Y.

    Convention matches official uv2backwardmap/fm2bm.m. Flat size is 448x448.
    """
    if uv is None or uv.ndim != 3 or uv.shape[2] < 3:
        raise ValueError("Expected a readable Doc3D UV EXR with at least 3 channels")
    valid = (uv[..., 0] > 0.5) & np.isfinite(uv[..., :3]).all(axis=-1)
    coords = np.stack([uv[..., 2], 1 - uv[..., 1]], axis=-1)
    valid &= ((coords >= -1e-4) & (coords <= 1.0001)).all(axis=-1)
    coords = np.clip(coords, 0, 1) * 448
    coords[~valid] = np.nan
    if valid.mean() < 0.01:
        raise ValueError("Doc3D UV has less than 1% valid page pixels")
    return coords


def save_sample(folder, image, mapping, meta, flat=None):
    folder.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(folder / "photo.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Could not write {folder}")
    np.save(folder / "gt_bmap_s2.npy", mapping.astype(np.float32))
    if flat is not None:
        cv2.imwrite(str(folder / "flat.png"), flat)
    # Write metadata last, so an interrupted sample is not considered complete.
    (folder / "gt.json").write_text(json.dumps({"meta": meta, "flat_ann": [], "photo_ann": []}, indent=2))


def prepare(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"Output must be empty to prevent stale split membership: {args.out}")
    rows = []
    if args.dataset == "deepscores":
        root = args.source
        if not (root / "deepscores_train.json").exists():
            root = root / "ds2_dense"
        for original in ("train", "test"):
            record = json.loads((root / f"deepscores_{original}.json").read_text())
            for im in record["images"]:
                name = im.get("filename", im.get("file_name"))
                if "-aug-" not in name:
                    raise ValueError(f"Unrecognized DeepScores score ID: {name}")
                group = name.split("-aug-", 1)[0]
                rows.append((root / "images" / name, group, original))
    else:
        # ZIP layouts may have an extra doc3d/ directory.
        roots = [p for p in args.source.rglob("img") if p.is_dir() and (p.parent / "uv").is_dir()]
        if len(roots) != 1:
            raise ValueError("Expected exactly one img/ directory next to uv/")
        root = roots[0].parent
        for path in sorted((root / "img").rglob("*.png")):
            if (root / "uv" / path.relative_to(root / "img").with_suffix(".exr")).exists():
                group = doc3d_group(path.stem)
                rows.append((path, group, None))
    if not rows:
        raise ValueError("No matched input samples found")
    # Random, deterministic selection instead of taking only the first score IDs.
    rows.sort(key=lambda r: str(r[0]))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(rows)
    selected = {s: [] for s in ("train", "val", "test")}
    for row in rows:
        split = split_for(row[1], args.seed)
        if not args.limit or len(selected[split]) < args.limit:
            selected[split].append(row)
    if any(not selected[s] for s in selected):
        raise ValueError("Too few source groups for nonempty train/val/test splits")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": args.dataset, "seed": args.seed, "limit_per_split": args.limit,
                "variants_train": args.variants if args.dataset == "deepscores" else 1,
                "source": str(args.source.resolve()),
                "photo_size": args.photo_size if args.dataset == "deepscores" else [448, 448],
                "flat_long_edge": args.flat_long_edge if args.dataset == "deepscores" else 448,
                "strength": args.strength if args.dataset == "deepscores" else None,
                "split_rule": "sha256(seed:source_group), 80/10/10",
                "group_rule": "score ID before -aug-" if args.dataset == "deepscores" else "parentmeshA_parentmeshB",
                "samples": []}
    for split, items in selected.items():
        (args.out / split).mkdir(exist_ok=True)
        for i, (path, group, original) in enumerate(items):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if args.dataset == "deepscores" else cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read {path}")
            sample_seed = int(hashlib.sha256(f"{args.seed}:{path.name}".encode()).hexdigest()[:16], 16)
            local_rng = np.random.default_rng(sample_seed)
            repeats = args.variants if split == "train" and args.dataset == "deepscores" else 1
            for variant in range(repeats):
                meta = {"source": args.dataset, "source_file": str(path.relative_to(root)),
                        "source_group": group, "original_split": original,
                        "supervision": {"seg": False, "struct": False, "kind": args.dataset == "deepscores"}}
                folder = args.out / split / f"{args.dataset}_{i:05d}_{variant:02d}"
                if args.dataset == "deepscores":
                    h, w = image.shape
                    scale = min(1, args.flat_long_edge / max(h, w))
                    flat = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
                    sim = simulate_photo(flat, [], local_rng, out_wh=tuple(args.photo_size), strength=args.strength)
                    meta.update(size=[flat.shape[1], flat.shape[0]], kind="staff")
                    save_sample(folder, sim["image"], sim["bmap_lo"], meta, flat)
                else:
                    uv_path = root / "uv" / path.relative_to(root / "img").with_suffix(".exr")
                    mapping = doc3d_uv_to_map(cv2.imread(str(uv_path), cv2.IMREAD_UNCHANGED))
                    if mapping.shape[:2] != image.shape[:2]:
                        raise ValueError(f"UV/image resolution mismatch: {path}")
                    meta.update(size=[448, 448], kind="unknown")
                    save_sample(folder, image, mapping[::2, ::2], meta)
                manifest["samples"].append({"path": str(folder.relative_to(args.out)), **meta})
            if (i + 1) % 10 == 0 or i + 1 == len(items):
                print(split, f"{i+1}/{len(items)} pages", flush=True)
    manifest["counts"] = {s: sum(x["path"].startswith(s + "/") for x in manifest["samples"]) for s in selected}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Prepared", manifest["counts"], "in", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=["deepscores", "doc3d"])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=64, help="source pages per split; 0 = all")
    ap.add_argument("--variants", type=int, default=2, help="warps per DeepScores training page")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--photo-size", type=int, nargs=2, default=[768, 1024], metavar=("W", "H"))
    ap.add_argument("--flat-long-edge", type=int, default=1200)
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()
    if args.limit < 0 or args.variants < 1 or min(args.photo_size) < 32 or args.flat_long_edge < 32:
        ap.error("Invalid limit, variant count or image dimensions")
    prepare(args)


if __name__ == "__main__":
    main()
