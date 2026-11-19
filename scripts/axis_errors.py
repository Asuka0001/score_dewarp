"""Per-axis decomposition of the dewarping error on a Doc3D score bench.

For each page and method the photo grid points are mapped to the output and
aligned to the flat page exactly as ``bench.py`` does (rotation + per-axis
scale + translation on the content box). Then:

  ad_x, ad_y    RMS residual per axis (% page width)
  sx, sy        RMS of log local stretch along x / y of the output relative to
                the flat page (%): local Jacobian E = d(output)/d(flat), from
                central differences over ``--step`` grid cells; E = I is perfect
  shx, shy      RMS of E[1,0] (horizontal lines bend / tilt) and E[0,1]
                (vertical lines lean), in %

Raw grids are saved per page (``grids/<sample>/<method>.npz``) for later
analysis without rerunning the pipeline.

  python scripts/axis_errors.py --data data/bench/doc3d_score/dev --methods input gcs metric \
      --out outputs/bench/axis_20261119
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench  # noqa: E402
from scoredewarp.eval.geometry import map_points, sample_geometry  # noqa: E402


def jac_errors(aligned, target, keep, step):
    """Local Jacobian E = dA/dT per grid cell from photo-grid differences."""
    s = step
    ax = (aligned[s:-s, 2 * s:] - aligned[s:-s, :-2 * s])   # d/dphoto_x
    ay = (aligned[2 * s:, s:-s] - aligned[:-2 * s, s:-s])   # d/dphoto_y
    tx = (target[s:-s, 2 * s:] - target[s:-s, :-2 * s])
    ty = (target[2 * s:, s:-s] - target[:-2 * s, s:-s])
    k = (keep[s:-s, 2 * s:] & keep[s:-s, :-2 * s] & keep[2 * s:, s:-s] & keep[:-2 * s, s:-s] & keep[s:-s, s:-s])
    A = np.stack([ax, ay], -1)          # (..., 2 out comps, 2 photo dirs)
    T = np.stack([tx, ty], -1)
    det = T[..., 0, 0] * T[..., 1, 1] - T[..., 0, 1] * T[..., 1, 0]
    k &= np.isfinite(A).all((-1, -2)) & np.isfinite(T).all((-1, -2)) & (np.abs(det) > 1e-12)
    A, T, det = A[k], T[k], det[k]
    Ti = np.stack([np.stack([T[:, 1, 1], -T[:, 0, 1]], -1), np.stack([-T[:, 1, 0], T[:, 0, 0]], -1)], -2) / det[:, None, None]
    E = A @ Ti
    ok = (E[:, 0, 0] > 0.05) & (E[:, 1, 1] > 0.05)
    E = E[ok]
    if len(E) < 20:
        return {}
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)) * 100)  # noqa: E731
    return {"sx": rms(np.log(E[:, 0, 0])), "sy": rms(np.log(E[:, 1, 1])),
            "shx": rms(E[:, 1, 0]), "shy": rms(E[:, 0, 1])}


def run_one(args):
    folder, methods, out, step = args
    bench.register_new_methods()
    folder = Path(folder)
    gt = json.loads((folder / "gt.json").read_text())
    meta = gt["meta"]
    image, _, points, target, valid = sample_geometry(folder, max_points=6000)
    aniso = meta.get("aspect_meaningful", True) is False
    with np.errstate(invalid="ignore"):
        content = bench.content_mask(gt, target)
    if content is None:
        content = np.ones_like(valid)
    rows = []
    for name in methods:
        row = {"sample": folder.name, "method": name, "kind": meta.get("kind"), "tier": meta.get("tier")}
        t0 = time.time()
        try:
            fn = bench.get_method(name)
            res = None if fn is None else fn(image)
            if res is None:
                pred, keep = points.astype(float), valid.copy()
            else:
                pred, keep = map_points(res.bmap, points)
            row["time"] = time.time() - t0
            keep &= valid & content
            if keep.sum() < 20:
                raise ValueError("too few content points")
            aligned = np.full(pred.shape, np.nan)
            aligned[keep] = bench._align(pred[keep], target[keep], aniso)
            r = aligned[keep] - target[keep]
            row["ad_c"] = float(np.sqrt((r ** 2).sum(-1).mean()) * 100)
            row["ad_x"] = float(np.sqrt((r[:, 0] ** 2).mean()) * 100)
            row["ad_y"] = float(np.sqrt((r[:, 1] ** 2).mean()) * 100)
            row.update(jac_errors(aligned, target, keep, step))
            d = Path(out) / "grids" / folder.name
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{name}.npz", pred=pred.astype(np.float32), target=target.astype(np.float32),
                                keep=keep, aligned=aligned.astype(np.float32))
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=["input", "gcs", "metric"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter", default="", help="regex on the sample folder name")
    args = ap.parse_args()
    folders = sorted(p.parent for p in args.data.rglob("gt.json"))
    if args.filter:
        import re
        folders = [f for f in folders if re.search(args.filter, f.name)]
    if args.limit:
        folders = folders[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), args.methods, str(args.out), args.step) for f in folders]
    rows, t0 = [], time.time()
    from multiprocessing import get_context
    with get_context("spawn").Pool(args.workers) as pool:
        for i, rs in enumerate(pool.imap_unordered(run_one, jobs)):
            rows.extend(rs)
            if (i + 1) % 10 == 0:
                print(f"{i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    with open(args.out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"done {time.time()-t0:.0f}s, errors {sum('error' in r for r in rows)}")


if __name__ == "__main__":
    main()
