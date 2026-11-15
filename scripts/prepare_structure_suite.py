"""Create a frozen, stratified music benchmark with disjoint content per split.

Each sample has an independent seed, source group and SHA-256 checksums. Test
is reserved until both ablations have finished; no adaptive threshold tuning.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.synth.distort import simulate_photo
from scoredewarp.synth.render import render_page
from scoredewarp.synth.fonts import cjk_font_paths, digit_font_paths, symbol_font_paths

KINDS = ("staff", "jianpu", "mixed")
PROFILES = {
    "mild": {"strength": .5},
    "standard": {"strength": 1.},
    "strong": {"strength": 1.3},
    "shadow": {"strength": 1., "shadow_probability": 1.},
    "blur": {"strength": 1., "blur_sigma": 1.8},
}


def generate(job):
    root, split, kind, profile, index, seed, photo_size = job
    cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    flat, ann, meta = render_page(kind, rng)
    sim = simulate_photo(flat, ann, rng, out_wh=photo_size, require_invertible=True, **PROFILES[profile])
    name = f"{kind}_{profile}_{index:03d}"
    group = f"structure-suite-{seed}"
    supervision = {head: True for head in ("seg", "struct", "kind")}
    meta.update(source="structure_suite", source_group=group, split=split,
                profile=profile, seed=seed, simulation=PROFILES[profile], supervision=supervision,
                projection=sim["projection"])
    path = Path(root) / split / name
    path.mkdir(parents=True)
    if not cv2.imwrite(str(path / "flat.png"), flat):
        raise RuntimeError(f"Cannot write {path}")
    if not cv2.imwrite(str(path / "photo.jpg"), sim["image"], [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Cannot write {path}")
    np.save(path / "gt_bmap_s2.npy", sim["bmap_lo"].astype(np.float16))
    (path / "gt.json").write_text(json.dumps({"meta": meta, "flat_ann": ann, "photo_ann": sim["ann"]}))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir())}
    return {"path": f"{split}/{name}", "source_group": group, "kind": kind,
            "profile": profile, "seed": seed, "size": meta["size"], "projection": sim["projection"],
            "supervision": supervision, "sha256": hashes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026111501)
    ap.add_argument("--train-per-cell", type=int, default=20)
    ap.add_argument("--val-per-cell", type=int, default=6)
    ap.add_argument("--test-per-cell", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--photo-size", nargs=2, type=int, default=[768, 1024])
    args = ap.parse_args()
    if min(args.train_per_cell, args.val_per_cell, args.test_per_cell, args.workers, *args.photo_size) < 1:
        ap.error("Counts, workers and image dimensions must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("Use a fresh empty output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    jobs, counts = [], {}
    for si, (split, count) in enumerate((("train", args.train_per_cell), ("val", args.val_per_cell),
                                       ("test", args.test_per_cell))):
        counts[split] = count * len(KINDS) * len(PROFILES)
        for ki, kind in enumerate(KINDS):
            for pi, profile in enumerate(PROFILES):
                for index in range(count):
                    seed = int(np.random.SeedSequence([args.seed, si, ki, pi, index]).generate_state(1, dtype=np.uint64)[0])
                    jobs.append((str(args.out), split, kind, profile, index, seed, tuple(args.photo_size)))
    # Save the complete protocol before generating or evaluating any sample.
    protocol = {"seed": args.seed, "profiles": PROFILES, "kinds": KINDS, "counts": counts,
                "photo_size": args.photo_size, "workers": args.workers,
                "projection_filter": {"grid": [97, 129], "min_signed_area_ratio": .02, "max_attempts": 64},
                "fonts": {name: [{"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                                 for p in paths] for name, paths in
                          (("digits", digit_font_paths()), ("cjk", cjk_font_paths()), ("symbols", symbol_font_paths()))}}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(generate, jobs):
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"generated {len(rows)}/{len(jobs)} {row['path']}", flush=True)
    for filename in ("flat.png", "photo.jpg"):
        if len({r["sha256"][filename] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate content: {filename}")
    (args.out / "manifest.json").write_text(json.dumps({**protocol, "samples": rows}, indent=2))
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
