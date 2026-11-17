"""Score-dewarping benchmark: ground-truth geometry + structure metrics.

Every number is computed from ground truth (dense photo->flat map and flat
annotations), never from our own detector.

Geometry (DocAligner / PaperEdge-style AD and LD, but with exact GT instead of
a flow estimate). Points on a regular grid over the visible page are pushed
through the method's photo->output map and aligned to their flat positions:
  ad      RMS residual after the best global alignment, % of page width
          (rotation + per-axis scale + translation when the dataset's flat
          aspect is not physical, e.g. Doc3D re-textures; similarity otherwise)
  ld      local distortion: error of neighbouring-point differences, % of the
          grid step (sensitive to local stretch / shear / wobble)
  cover   fraction of visible page points present in the output
Structure (from flat labels mapped through the method):
  straight   staff-line y std / spacing      spacing_cv  staff spacing CV
  vtilt      barline + stem angle from vertical (deg)
  jp_base    jianpu baseline std / digit height, jp_hcv, jp_tilt

  python scripts/bench.py --data data/bench/doc3d_score/dev --methods abc ab --out outputs/bench/dev
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.eval.geometry import map_points, sample_geometry  # noqa: E402
from scoredewarp.eval.metrics import _ang_from_vertical, _similarity_fit, evaluate  # noqa: E402

cv2.setNumThreads(1)


# ---------------------------------------------------------------- methods ----
def _abc(img, **kw):
    from scoredewarp.core.pipeline import Config, dewarp
    return dewarp(img, Config(**kw))


METHODS = {
    "input": None,                                   # identity: the photo itself
    "abc": lambda img: _abc(img),                    # legacy A+B+C (default CLI before rework)
    "ab": lambda img: _abc(img, use_field=False),    # legacy A+B
    "abc_mesh": lambda img: _document(img),          # legacy ABC + staff line mesh ("document")
}


def _document(img):
    from scoredewarp.core.pipeline import dewarp_document
    return dewarp_document(img)


def register_new_methods():
    try:
        from scoredewarp.core import metric_pipeline as mp
    except ImportError:
        return
    for name, fn in mp.BENCH_METHODS.items():
        METHODS[name] = fn
    METHODS["__spec__"] = mp.method_from_spec


def get_method(name):
    if name not in METHODS and name.startswith("gcs:"):
        METHODS[name] = METHODS["__spec__"](name)
    return METHODS[name]


# ---------------------------------------------------------------- metrics ----
def _align(pred, target, aniso):
    """Return pred aligned to target (rotation [+ per-axis scale] + translation)."""
    s, R, t = _similarity_fit(pred, target)
    if not aniso:
        return s * pred @ R.T + t
    th0 = np.arctan2(R[1, 0], R[0, 0])

    def solve(th):
        c, s_ = np.cos(th), np.sin(th)
        q = pred @ np.array([[c, -s_], [s_, c]]).T
        out = np.empty_like(q)
        for k in (0, 1):
            A = np.stack([q[:, k], np.ones(len(q))], 1)
            coef, *_ = np.linalg.lstsq(A, target[:, k], rcond=None)
            out[:, k] = A @ coef
        return out

    best = minimize_scalar(lambda th: ((solve(th) - target) ** 2).sum(), bounds=(th0 - 0.2, th0 + 0.2),
                           method="bounded", options={"xatol": 1e-5})
    return solve(best.x)


def _ad_ld(pred, target, keep, aniso):
    if keep.sum() < 20:
        return {}
    aligned = np.full(pred.shape, np.nan)
    aligned[keep] = _align(pred[keep], target[keep], aniso)
    out = {"ad": float(np.sqrt(((aligned[keep] - target[keep]) ** 2).sum(-1).mean()) * 100)}
    err, ref = [], []
    for axis in (0, 1):
        a = np.diff(aligned, axis=axis)
        b = np.diff(target, axis=axis)
        ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
        err.append(((a - b)[ok] ** 2).sum(-1))
        ref.append((b[ok] ** 2).sum(-1))
    err, ref = np.concatenate(err), np.concatenate(ref)
    if len(ref):
        out["ld"] = float(np.sqrt(err.mean() / ref.mean()) * 100)
    return out


def geometry(points, target, valid, bmap, aniso, content=None):
    """Whole visible page (ad, ld) and the printed-content box only (ad_c, ld_c):
    a method that crops the margins is not rewarded by dropping its worst points,
    and content errors are not diluted by blank margins."""
    if bmap is None:
        pred, keep = points.astype(float), valid.copy()
    else:
        pred, keep = map_points(bmap, points)
    keep &= valid
    n = int(valid.sum())
    out = {"cover": float(keep.sum() / max(n, 1))}
    out.update(_ad_ld(pred, target, keep, aniso))
    if content is not None:
        inc = valid & content
        out["cover_c"] = float((keep & inc).sum() / max(inc.sum(), 1))
        out.update({k + "_c": v for k, v in _ad_ld(pred, target, keep & inc, aniso).items()})
    return out


def content_mask(gt, target):
    """Grid points inside the bounding box of all flat annotations."""
    pts = [np.asarray(a["pts"], float).reshape(-1, 2) for a in gt["flat_ann"] if a.get("kind") != "region"]
    if not pts:
        return None
    pts = np.concatenate(pts) / gt["meta"]["size"][0]
    lo, hi = np.nanmin(pts, 0), np.nanmax(pts, 0)
    return (target[..., 0] >= lo[0]) & (target[..., 0] <= hi[0]) & (target[..., 1] >= lo[1]) & (target[..., 1] <= hi[1])


def structure(gt, bmap):
    if bmap is None:
        fwd, size = (lambda p: np.asarray(p, float)), None
    else:
        fwd, size = bmap.forward_points, bmap.out_size
    m = evaluate(gt, fwd, out_size=size)
    out = {k: m[k] for k in ("staff_straight", "staff_spacing_cv", "jp_baseline", "jp_height_cv", "jp_tilt_deg")
           if k in m and m[k] is not None}
    tilts = []
    for a in gt["photo_ann"]:
        if a["kind"] in ("stem", "barline"):
            p = np.asarray(a["pts"], float)[[0, -1]]
            if not np.isfinite(p).all():
                continue
            q = np.asarray(fwd(p), float)
            if size is not None:
                q[~((q >= 0).all(1) & (q < np.asarray(size)).all(1))] = np.nan
            if np.isfinite(q).all():
                tilts.append(_ang_from_vertical(q[:1], q[1:])[0])
    if tilts:
        out["vtilt"] = float(np.mean(tilts))
    return out


def run_one(args):
    folder, methods, save_dir = args
    folder = Path(folder)
    gt = json.loads((folder / "gt.json").read_text())
    meta = gt["meta"]
    image, _, points, target, valid = sample_geometry(folder, max_points=6000)
    aniso = meta.get("aspect_meaningful", True) is False
    with np.errstate(invalid="ignore"):
        content = content_mask(gt, target)
    rows = []
    for name in methods:
        row = {"sample": f"{folder.parent.name}/{folder.name}", "method": name, "kind": meta.get("kind"),
               "tier": meta.get("tier", meta.get("profile", "-")), "flat_source": meta.get("flat_source", "-")}
        t0 = time.time()
        try:
            fn = get_method(name)
            res = None if fn is None else fn(image)
            bmap = None if res is None else res.bmap
            row["time"] = time.time() - t0
            row.update(geometry(points, target, valid, bmap, aniso, content))
            row.update(structure(gt, bmap))
            if res is not None:
                row["out_size"] = list(bmap.out_size)
                st = getattr(res, "stages", {}) or {}
                row["notes"] = {k: st[k] for k in ("surface_rejected", "fallback", "model") if k in st}
                if save_dir:
                    d = Path(save_dir) / folder.parent.name / folder.name
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{name}.jpg"), res.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:  # a crash is a failure, not a missing row
            row["error"] = f"{type(e).__name__}: {e}"
            row["trace"] = traceback.format_exc()[-800:]
            row["cover"] = 0.0
        rows.append(row)
    return rows


# ---------------------------------------------------------------- report -----
KEYS = ["ad_c", "ld_c", "cover_c", "ad", "ld", "cover", "straight", "spacing_cv", "vtilt", "jp_base", "jp_hcv", "jp_tilt", "time"]
ALIAS = {"staff_straight": "straight", "staff_spacing_cv": "spacing_cv", "jp_baseline": "jp_base",
         "jp_height_cv": "jp_hcv", "jp_tilt_deg": "jp_tilt"}
LOWER_BETTER = {k: not k.startswith("cover") for k in KEYS}


def summarize(rows, methods):
    for r in rows:
        for a, b in ALIAS.items():
            if a in r:
                r[b] = r.pop(a)
    lines = []

    def table(title, subset):
        lines.append(f"\n### {title}\n")
        lines.append("| method | n | fail | " + " | ".join(KEYS) + " |")
        lines.append("|---|---:|---:|" + "---:|" * len(KEYS))
        for m in methods:
            rs = [r for r in subset if r["method"] == m]
            if not rs:
                continue
            fail = sum(1 for r in rs if "error" in r or r.get("cover", 0) < 0.5)
            cells = []
            for k in KEYS:
                v = [r[k] for r in rs if k in r and r[k] is not None and np.isfinite(r[k])]
                cells.append(f"{np.median(v):.3f}" if v else "—")
            lines.append(f"| {m} | {len(rs)} | {fail} | " + " | ".join(cells) + " |")

    table("all (median)", rows)
    for kind in sorted({r["kind"] for r in rows}):
        table(f"kind={kind}", [r for r in rows if r["kind"] == kind])
    for tier in sorted({r["tier"] for r in rows}):
        table(f"tier={tier}", [r for r in rows if r["tier"] == tier])
    if len(methods) > 1:
        base = methods[1] if methods[0] == "input" and len(methods) > 2 else methods[0]
        lines.append(f"\n### paired vs `{base}` (wins = strictly better; samples where both defined)\n")
        lines.append("| method | metric | n | wins | median diff |")
        lines.append("|---|---|---:|---:|---:|")
        by = {(r["sample"], r["method"]): r for r in rows}
        samples = sorted({r["sample"] for r in rows})
        for m in methods:
            if m == base:
                continue
            for k in ("ad_c", "ld_c", "cover_c", "ad", "straight", "vtilt", "jp_base"):
                d = []
                for s in samples:
                    a, b = by.get((s, m), {}), by.get((s, base), {})
                    if k in a and k in b and a[k] is not None and b[k] is not None:
                        d.append(a[k] - b[k])
                if d:
                    d = np.array(d)
                    wins = int((d < 0).sum() if LOWER_BETTER[k] else (d > 0).sum())
                    lines.append(f"| {m} | {k} | {len(d)} | {wins} | {np.median(d):+.4f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=["input", "abc"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter", default="", help="substring of sample name")
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()
    register_new_methods()
    for m in args.methods:
        if m not in METHODS and not m.startswith("gcs:"):
            raise SystemExit(f"unknown method {m}; have {sorted(METHODS)}")
    folders = sorted(p.parent for p in args.data.rglob("gt.json"))
    folders = [f for f in folders if args.filter in str(f)]
    if args.limit:
        folders = folders[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), args.methods, str(args.out / "images") if args.save_images else None) for f in folders]
    rows = []
    t0 = time.time()
    if args.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(args.workers, initializer=register_new_methods) as pool:
            for i, rs in enumerate(pool.imap_unordered(run_one, jobs)):
                rows.extend(rs)
                if (i + 1) % 10 == 0:
                    print(f"{i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    else:
        for i, j in enumerate(jobs):
            rows.extend(run_one(j))
            print(f"{i+1}/{len(jobs)}", flush=True)
    with open(args.out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    rep = f"# bench `{args.data}`  methods={args.methods}\n" + summarize(rows, args.methods)
    errs = [r for r in rows if "error" in r]
    if errs:
        rep += f"\n\n{len(errs)} errors, first: {errs[0]['sample']} {errs[0]['method']} {errs[0]['error']}"
    (args.out / "summary.md").write_text(rep)
    print(rep)


if __name__ == "__main__":
    main()
