"""Build the score-dewarping benchmark on real captured paper deformations.

Flat pages
  * DeepScoresV2 dense (original test split): real engraving variety. Staff
    lines are located exactly on the clean digital page (projection peaks
    inside each annotated staff box); stems and noteheads come from the boxes.
  * ``render_page`` staff / jianpu / mixed pages with full structure labels.
Deformations
  * Doc3D UV fields (real captured paper), split dev/test by parent mesh pair.
    Pages with < 85% of the sheet visible are skipped; each sample records a
    non-planarity score and a tier (mild / moderate / severe by tercile over
    the whole pool), so results can be stratified instead of filtered.

  python scripts/make_doc3d_bench.py --out data/bench/doc3d_score --per-split 120
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.synth.doc3d_retexture import read_doc3d_uv, render, uv_statistics  # noqa: E402
from scoredewarp.synth.render import render_page  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOC3D = ROOT / "data/external/doc3d"
DS = ROOT / "data/external/deepscores/ds2_dense"
STAFF_CATS, STEM_CATS = {"135", "208"}, {"42", "161"}


def doc3d_group(stem):
    mesh = stem.split("-", 1)[0].split("_")
    return "_".join(mesh[-2:])


def split_of(group, seed):
    return "dev" if int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 2 == 0 else "test"


def deepscores_pages():
    rec = json.loads((DS / "deepscores_test.json").read_text())
    cats = rec["categories"]
    head_cats = {k for k, v in cats.items() if v["name"].startswith("notehead")}
    anns = rec["annotations"]
    return rec["images"], anns, head_cats


def deepscores_flat(image_rec, anns, head_cats, long_edge=2000):
    """Return (flat gray, annotations in flat px) or None if staves are not clean."""
    gray = cv2.imread(str(DS / "images" / image_rec["filename"]), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    h, w = gray.shape
    s = min(1.0, long_edge / max(h, w))
    ann, staff_id = [], 0
    dark = 255.0 - gray.astype(np.float32)
    staves, stems, heads = [], [], []
    for aid in image_rec["ann_ids"]:
        a = anns[aid]
        cat = set(a["cat_id"]) if isinstance(a["cat_id"], list) else {a["cat_id"]}
        x0, y0, x1, y1 = a["a_bbox"]
        if cat & STAFF_CATS:
            staves.append((x0, y0, x1, y1))
        elif cat & STEM_CATS:
            stems.append((x0, y0, x1, y1))
        elif cat & head_cats:
            heads.append((x0, y0, x1, y1))
    if len(staves) < 2:
        return None
    for x0, y0, x1, y1 in sorted(set(staves), key=lambda b: b[1]):
        sp = (y1 - y0) / 4.0
        if sp < 4:
            continue
        r0, r1 = int(max(0, y0 - sp / 2)), int(min(h, y1 + sp / 2 + 1))
        prof = dark[r0:r1, int(x0):int(x1)].mean(1)
        ys = []
        for i in range(5):
            c = y0 + i * sp - r0
            lo, hi = int(max(0, c - sp / 3)), int(min(len(prof), c + sp / 3 + 1))
            seg = prof[lo:hi]
            if seg.max() < 60:  # no clear line: box is not a clean 5-line staff
                ys = None
                break
            wts = np.clip(seg - 0.5 * seg.max(), 0, None)
            ys.append(r0 + lo + float((wts * np.arange(len(seg))).sum() / wts.sum()))
        if ys is None:
            continue
        d = np.diff(ys)
        if d.min() < 0.8 * d.mean() or d.max() > 1.2 * d.mean():
            continue
        spacing = float(d.mean() * s)
        xs = np.arange(x0 * s, x1 * s, 10.0)
        for i, y in enumerate(ys):
            ann.append({"kind": "staff_line", "staff": staff_id, "line": i, "spacing": spacing,
                        "pts": [[float(x), float(y * s)] for x in xs]})
        staff_id += 1
    if staff_id < 2:
        return None
    sp_med = np.median([a["spacing"] for a in ann])
    for x0, y0, x1, y1 in stems:
        if (y1 - y0) * s > 2 * sp_med:
            cx = 0.5 * (x0 + x1) * s
            ann.append({"kind": "stem", "pts": [[cx, y0 * s], [cx, y1 * s]]})
    for x0, y0, x1, y1 in heads:
        ann.append({"kind": "notehead", "pts": [[0.5 * (x0 + x1) * s, 0.5 * (y0 + y1) * s]]})
    flat = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return flat, ann, {"staff_spacing": float(sp_med), "source_file": image_rec["filename"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/bench/doc3d_score")
    ap.add_argument("--per-split", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--size", type=int, default=1600)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"Output must be empty: {args.out}")
    rng = np.random.default_rng(args.seed)

    # ---- deformation pool with tiers -----------------------------------------
    pool = []
    for uv_path in sorted((DOC3D / "uv").rglob("*.exr")):
        img_path = DOC3D / "img" / uv_path.relative_to(DOC3D / "uv").with_suffix(".png")
        if not img_path.exists():
            continue
        coords, valid = read_doc3d_uv(uv_path)
        st = uv_statistics(coords, valid)
        if st["flat_coverage"] < 0.85 or st["page_fraction"] < 0.15:
            continue
        pool.append({"uv": uv_path, "img": img_path, "group": doc3d_group(uv_path.stem), **st})
    cuts = np.percentile([p["nonplanarity"] for p in pool], [100 / 3, 200 / 3])
    for p in pool:
        p["tier"] = ["mild", "moderate", "severe"][int(np.searchsorted(cuts, p["nonplanarity"]))]
        p["split"] = split_of(p["group"], args.seed)
    print(f"Doc3D pool {len(pool)} usable; tier cuts (nonplanarity %) {cuts.round(3).tolist()}", flush=True)

    images, anns, head_cats = deepscores_pages()
    order = rng.permutation(len(images))
    ds_iter = iter(order)
    kinds = ["deepscores", "staff", "jianpu", "mixed", "deepscores"]  # 40% real engraving
    manifest = {"seed": args.seed, "size": args.size, "tier_cuts": cuts.tolist(), "samples": []}
    for split in ("dev", "test"):
        cand = [p for p in pool if p["split"] == split]
        pick = rng.permutation(len(cand))[:args.per_split]
        for i, j in enumerate(pick):
            p = cand[j]
            kind = kinds[i % len(kinds)]
            sample_rng = np.random.default_rng([args.seed, i, 0 if split == "dev" else 1])
            if kind == "deepscores":
                res = None
                while res is None:
                    res = deepscores_flat(images[next(ds_iter)], anns, head_cats)
                flat, flat_ann, extra = res
                meta = {"kind": "staff", "flat_source": "deepscores", **extra}
            else:
                flat, flat_ann, m = render_page(kind, sample_rng)
                meta = {"kind": kind, "flat_source": "render", "staff_spacing": m["staff_spacing"],
                        "digit_height": m["digit_height"]}
            photo, bmap, photo_ann, st = render(flat, flat_ann, p["uv"], p["img"], sample_rng, args.size)
            name = f"{split}/{i:04d}_{meta['flat_source'][:2]}_{meta['kind']}_{p['tier']}"
            folder = args.out / name
            folder.mkdir(parents=True)
            cv2.imwrite(str(folder / "photo.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(folder / "flat.png"), flat)
            np.save(folder / "gt_bmap_s2.npy", bmap)
            meta.update(size=[flat.shape[1], flat.shape[0]], source="doc3d_retexture", split=split,
                        doc3d=str(p["uv"].relative_to(DOC3D)), doc3d_group=p["group"], tier=p["tier"],
                        nonplanarity=p["nonplanarity"], flat_coverage=p["flat_coverage"],
                        aspect_meaningful=False)
            (folder / "gt.json").write_text(json.dumps({"meta": meta, "flat_ann": flat_ann, "photo_ann": photo_ann}))
            manifest["samples"].append({"path": name, **{k: meta[k] for k in ("kind", "flat_source", "tier", "nonplanarity")}})
            if (i + 1) % 20 == 0:
                print(split, i + 1, flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("done", len(manifest["samples"]))


if __name__ == "__main__":
    main()
