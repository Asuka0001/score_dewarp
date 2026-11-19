"""Text-block side straightness from saved grids (``axis_errors.py`` output).

The left / right ends of every staff and jianpu row lie on two vertical lines
of the flat page (a justified score). GT ends are mapped through a method's
saved grid (photo -> output, already aligned to the flat page) and compared
with their flat positions; per side:

  edge_bend   RMS of the x error after removing a straight line in y
              (% page width): the side is bent / wavy in the output
  edge_slant  |angle| of that straight line (deg): the side is slanted

  python scripts/edge_metrics.py outputs/bench/axis_20261119 metric [more_dirs:method ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parents[1] / "data/bench/doc3d_score/dev"


def gt_ends(gt):
    """(left (N,2), right (N,2)) GT row ends on the flat page, units of page width."""
    W = gt["meta"]["size"][0]
    staff = defaultdict(list)
    rows = defaultdict(list)
    bars = []
    for a in gt["flat_ann"]:
        p = np.asarray(a["pts"], float).reshape(-1, 2) / W
        if a["kind"] == "staff_line":
            staff[a["staff"]].append(p)
        elif a["kind"] == "jp_char":
            rows[a["row"]].append(p)
        elif a["kind"] == "barline":
            bars.append(p)
    L, R = [], []
    for ps in staff.values():
        q = np.concatenate(ps)
        y = q[:, 1].mean()
        L.append((q[:, 0].min(), y))
        R.append((q[:, 0].max(), y))
    for ps in rows.values():
        q = np.concatenate(ps)
        y0, y1 = q[:, 1].min(), q[:, 1].max()
        xr = q[:, 0].max()
        for b in bars:
            if y0 - 0.01 <= b[:, 1].mean() <= y1 + 0.01:
                xr = max(xr, b[:, 0].max())
        L.append((q[:, 0].min(), (y0 + y1) / 2))
        R.append((xr, (y0 + y1) / 2))
    return np.array(L).reshape(-1, 2), np.array(R).reshape(-1, 2)


def page_edges(npz, gt):
    z = np.load(npz)
    keep = z["keep"]
    T, A_ = z["target"][keep].astype(float), z["aligned"][keep].astype(float)
    lin = LinearNDInterpolator(T, A_)
    tree = cKDTree(T)

    def f(P):
        """Linear interpolation; ends on the content-box border (outside the
        hull of kept grid points) by a local affine fit to the 12 nearest."""
        Q = lin(P)
        for i in np.flatnonzero(~np.isfinite(Q).all(1)):
            _, idx = tree.query(P[i], 12)
            M = np.c_[T[idx], np.ones(len(idx))]
            c, *_ = np.linalg.lstsq(M, A_[idx], rcond=None)
            Q[i] = np.r_[P[i], 1.0] @ c
        return Q
    out = {}
    for name, P in zip(("left", "right"), gt_ends(gt)):
        if len(P) < 3:
            continue
        Q = f(P)
        ok = np.isfinite(Q).all(1)
        if ok.sum() < 3:
            continue
        d = Q[ok, 0] - P[ok, 0]
        y = P[ok, 1]
        A = np.stack([y, np.ones_like(y)], 1)
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        out[name] = (float(np.sqrt(np.mean((d - A @ coef) ** 2)) * 100), float(abs(np.degrees(np.arctan(coef[0])))))
    if not out:
        return None
    return {"edge_bend": float(np.mean([v[0] for v in out.values()])),
            "edge_slant": float(np.mean([v[1] for v in out.values()]))}


def collect(run_dir, method):
    res = {}
    for d in sorted((Path(run_dir) / "grids").iterdir()):
        f = d / f"{method.replace('/', '_')}.npz"
        if not f.exists():
            continue
        gt = json.loads((DATA / d.name / "gt.json").read_text())
        e = page_edges(f, gt)
        if e:
            res[d.name] = e
    return res


if __name__ == "__main__":
    runs = [a.split(":", 1) if ":" in a else (a, None) for a in sys.argv[1:]]
    base = collect(sys.argv[1], sys.argv[2])
    print(f"base {sys.argv[1]} {sys.argv[2]}: {len(base)} pages")
    for kind in ("all", "staff", "jianpu", "mixed"):
        ks = [s for s in base if kind == "all" or kind in s]
        print(f"  {kind:6s} n={len(ks):3d} edge_bend median {np.median([base[s]['edge_bend'] for s in ks]):.3f} "
              f"mean {np.mean([base[s]['edge_bend'] for s in ks]):.3f} | edge_slant median "
              f"{np.median([base[s]['edge_slant'] for s in ks]):.3f} deg")
