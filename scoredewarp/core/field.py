"""Stage C: non-rigid residual field (sparse linear least squares with IRLS).

Forward map  F(x, y) = (U(x, y), V(x, y)), bilinear on a coarse node grid.
All structural priors become *linear* difference constraints on U or V:

  V along a staff line / underline / text baseline / topline  : equal
  V between adjacent staff lines                              : = staff spacing  (五线谱)
  V(bottom) - V(top) of a glyph                               : = glyph height   (简谱/文本)
  U along a horizontal structure                              : = scaled arc length
  U along a barline / stem, U(top) vs U(bottom) of a glyph    : equal (vertical / upright)
plus second-order smoothness and a weak identity prior. Handles local bumps,
creases and residual errors the low-order 3-D model cannot represent.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.linalg import spsolve

from .structures import Structures
from .transforms import BackwardMap


class _Grid:
    def __init__(self, W, H, node):
        self.node = node
        self.nx = int(np.ceil(W / node)) + 1
        self.ny = int(np.ceil(H / node)) + 1
        self.n = self.nx * self.ny

    def weights(self, pts):
        """Sparse (N, n) bilinear interpolation matrix."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        gx = np.clip(pts[:, 0] / self.node, 0, self.nx - 1 - 1e-6)
        gy = np.clip(pts[:, 1] / self.node, 0, self.ny - 1 - 1e-6)
        i0, j0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
        fx, fy = gx - i0, gy - j0
        N = len(pts)
        rows = np.repeat(np.arange(N), 4)
        cols = np.stack([j0 * self.nx + i0, j0 * self.nx + i0 + 1, (j0 + 1) * self.nx + i0, (j0 + 1) * self.nx + i0 + 1], 1).ravel()
        vals = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1).ravel()
        return csr_matrix((vals, (rows, cols)), shape=(N, self.n))

    def smooth_ops(self):
        idx = np.arange(self.n).reshape(self.ny, self.nx)
        ops = []
        for a, b, c in [(idx[:, :-2], idx[:, 1:-1], idx[:, 2:]), (idx[:-2], idx[1:-1], idx[2:])]:
            a, b, c = a.ravel(), b.ravel(), c.ravel()
            m = len(a)
            r = np.repeat(np.arange(m), 3)
            ops.append(coo_matrix((np.tile([1.0, -2.0, 1.0], m), (r, np.stack([a, b, c], 1).ravel())), shape=(m, self.n)))
        a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        m = len(a)
        r = np.repeat(np.arange(m), 4)
        ops.append(coo_matrix((np.tile([1.0, -1.0, -1.0, 1.0], m), (r, np.stack([a, b, c, d], 1).ravel())), shape=(m, self.n)))
        return vstack(ops).tocsr()


class _System:
    def __init__(self, grid):
        self.g = grid
        self.A, self.b, self.w = [], [], []

    def diff(self, pa, pb, target, w):
        """rows: F(pa) - F(pb) = target."""
        if len(pa) == 0:
            return
        self.A.append(self.g.weights(pa) - self.g.weights(pb))
        self.b.append(np.broadcast_to(np.asarray(target, float), (len(pa),)).copy())
        self.w.append(np.broadcast_to(np.asarray(w, float), (len(pa),)).copy())

    def solve(self, smooth, lam, ident, mu, iters=4, delta=2.0):
        A = vstack(self.A).tocsr()
        b = np.concatenate(self.b)
        w0 = np.concatenate(self.w)
        n = self.g.n
        Rs = smooth * lam
        I = csr_matrix((np.full(n, mu), (np.arange(n), np.arange(n))), shape=(n, n))
        rob = np.ones_like(b)
        x = ident.copy()
        for _ in range(iters):
            ww = w0 * rob
            Aw = A.multiply(ww[:, None]).tocsr()
            M = (Aw.T @ Aw + Rs.T @ Rs + I.T @ I).tocsc()
            rhs = Aw.T @ (b * ww) + I.T @ (mu * ident)
            x = spsolve(M, rhs)
            r = np.abs(A @ x - b)
            rob = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9))
        self.resid = A @ x - b
        return x


def _sub(pts, d):
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= d:
            keep.append(i)
    if keep[-1] != len(pts) - 1:
        keep.append(len(pts) - 1)
    return pts[keep]


def build_constraints(S: Structures, grid, priors=("staff", "jianpu", "text", "vertical")):
    U, V = _System(grid), _System(grid)
    staves = S.staves if "staff" in priors else []
    s_all = [st.spacing for st in staves]
    s_ref = float(np.median(s_all)) if s_all else None
    for st in staves:
        s_t = st.spacing if (s_ref is None or abs(st.spacing - s_ref) > 0.2 * s_ref) else s_ref
        T = np.array(st.tuples)  # (n,5,2)
        if len(T) == 0:
            continue
        loc = np.diff(T[:, :, 1], axis=1).mean(1)          # local spacing per tuple
        for i in range(4):
            V.diff(T[:, i + 1], T[:, i], s_t, 1.0)
            U.diff(T[:, i + 1], T[:, i], 0.0, 0.3)
        xs_t = T[:, 0, 0]
        order = np.argsort(xs_t)
        for fi in st.frag_ids:
            p = _sub(S.fragments[fi].pts, 20)
            if len(p) < 2:
                continue
            V.diff(p[1:], p[:-1], 0.0, 1.0)
            sc = s_t / np.interp((p[1:, 0] + p[:-1, 0]) / 2, xs_t[order], loc[order])
            U.diff(p[1:], p[:-1], np.linalg.norm(np.diff(p, axis=0), axis=1) * sc, 0.5)
    for f in S.fragments:
        if f.staff >= 0 and "staff" in priors:
            continue
        if f.staff < 0 and "jianpu" not in priors and "text" not in priors:
            continue
        p = _sub(f.pts, 20)
        if len(p) < 2 or np.linalg.norm(p[-1] - p[0]) < 40:
            continue
        V.diff(p[1:], p[:-1], 0.0, 0.4)
    jp = [r for r in S.rows if r.kind == "jianpu" and len(r.full) >= 3]
    h_ref = float(np.median([r.height for r in jp])) if jp else None
    for r in S.rows:
        if len(r.full) < 3:
            continue
        if r.kind == "jianpu" and "jianpu" not in priors:
            continue
        if r.kind == "text" and "text" not in priors:
            continue
        fb = [r.blobs[i] for i in r.full]
        top = np.array([[b.cx, b.y0] for b in fb])
        bot = np.array([[b.cx, b.y1] for b in fb])
        mid = (top + bot) / 2
        hs = bot[:, 1] - top[:, 1]
        h_t = h_ref if (r.kind == "jianpu" and h_ref and abs(r.height - h_ref) < 0.2 * h_ref) else r.height
        w = 0.8 if r.kind == "jianpu" else 0.5
        V.diff(top[1:], top[:-1], 0.0, w)
        V.diff(bot[1:], bot[:-1], 0.0, w)
        V.diff(bot, top, h_t, w)
        hloc = np.convolve(np.pad(hs, 2, mode="edge"), np.ones(5) / 5, "valid")
        sc = h_t / ((hloc[1:] + hloc[:-1]) / 2)
        U.diff(mid[1:], mid[:-1], np.linalg.norm(np.diff(mid, axis=0), axis=1) * sc, 0.3 * w)
    if "vertical" in priors:
        for v in S.vsegs:
            p = _sub(v.pts, 15)
            if len(p) >= 2:
                U.diff(p[1:], p[:-1], 0.0, 0.6)
    return U, V


def fit_field(S: Structures, node=48, lam=6.0, mu=0.02, priors=("staff", "jianpu", "text", "vertical")):
    H, W = S.shape
    g = _Grid(W, H, node)
    U, V = build_constraints(S, g, priors)
    if not U.A or not V.A:
        return None
    gx, gy = np.meshgrid(np.arange(g.nx) * node, np.arange(g.ny) * node)
    sm = g.smooth_ops()
    u = U.solve(sm, lam, gx.ravel().astype(float), mu)
    v = V.solve(sm, lam, gy.ravel().astype(float), mu)
    UV = np.stack([u, v], 1).reshape(g.ny, g.nx, 2)
    # recentre so that the content stays on the same canvas
    UV -= (UV - np.stack([gx, gy], -1)).reshape(-1, 2).mean(0)

    # forward-map a dense set of source points, invert by scattered interpolation
    st = 8
    sx, sy = np.meshgrid(np.arange(0, W + st, st, dtype=float), np.arange(0, H + st, st, dtype=float))
    src = np.stack([sx.ravel(), sy.ravel()], 1)
    Wt = g.weights(src)
    dst = np.stack([Wt @ UV[..., 0].ravel(), Wt @ UV[..., 1].ravel()], 1)
    inv = LinearNDInterpolator(dst, src)

    def fn(q):
        r = inv(q)
        return r

    bm = BackwardMap.from_function(fn, (W, H), stride=4)
    bm.forward_nodes = UV
    return bm
