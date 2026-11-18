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

    def mean_gradient(self, axis, value, w):
        """one row: mean forward difference of F along ``axis`` (1 = x, 0 = y)
        over the whole grid = ``value`` (gauge: pins the global scale)."""
        idx = np.arange(self.g.n).reshape(self.g.ny, self.g.nx)
        a, b = (idx[:, 1:], idx[:, :-1]) if axis == 1 else (idx[1:], idx[:-1])
        m = a.size
        row = coo_matrix((np.concatenate([np.full(m, 1.0 / m), np.full(m, -1.0 / m)]),
                          (np.zeros(2 * m, int), np.concatenate([a.ravel(), b.ravel()]))), shape=(1, self.g.n))
        self.A.append(row.tocsr())
        self.b.append(np.array([float(value)]))
        self.w.append(np.array([float(w)]))

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


def _row_ends(S: Structures, side, reject=1.5):
    """One end point per staff row, outliers removed: (N, 2) array.

    A row's end is where its staff lines end: per line the outermost fragment
    end (staves the detector split into pieces at one height form one row),
    then the line with the median x (one line running on into a hairpin or cut
    short by a clef does not move it). Not the 5-line tuples: notes and clefs
    interrupt those, so they stop up to several spacings short of the ends.
    Rows more than ``reject`` spacings off a robust line through all ends are
    dropped: an indented first system, a short last staff of an exercise."""
    from scipy.stats import theilslopes
    from .gcs import _staff_rows
    row = _staff_rows(S.staves)
    ext = {}
    for k, st in enumerate(S.staves):
        for fi in st.frag_ids:
            f = S.fragments[fi]
            if len(f.pts) < 2 or f.line < 0:
                continue
            p = f.pts[0] if side == 0 else f.pts[-1]
            key = (row[k], f.line)
            if key not in ext or (p[0] < ext[key][0] if side == 0 else p[0] > ext[key][0]):
                ext[key] = p
    by_row = {}
    for (r, _), p in ext.items():
        by_row.setdefault(r, []).append(p)
    P = []
    for r in sorted(by_row):
        q = np.array(by_row[r])
        if len(q) >= 3:
            P.append(q[np.argsort(q[:, 0])[len(q) // 2]])
    if len(P) < 3:
        return np.zeros((0, 2))
    P = np.array(P)
    sp = float(np.median([st.spacing for st in S.staves]))
    k, c = theilslopes(P[:, 0], P[:, 1])[:2]
    ok = np.abs(P[:, 0] - (c + k * P[:, 1])) < reject * sp
    return P[ok] if ok.sum() >= 3 else np.zeros((0, 2))


def margin_stats(S: Structures, min_rows=3):
    """Text-block margins of a (corrected) page from its staves, or None:
    left / right: (x at the top row, x at the bottom row, slant in deg; + = the
    margin runs right going down), keystone: bottom / top staff width - 1,
    lines: side -> (slope, intercept) of x = intercept + slope * y, y: (top, bottom)
    row, n: rows used per side."""
    out, lines, ys, n = {}, {}, [], {}
    for side, name in ((0, "left"), (-1, "right")):
        P = _row_ends(S, side)
        if len(P) < min_rows or np.ptp(P[:, 1]) <= 0:
            return None
        k, c = np.polyfit(P[:, 1], P[:, 0], 1)
        lines[name], n[name] = (float(k), float(c)), len(P)
        ys += [P[:, 1].min(), P[:, 1].max()]
    y0, y1 = min(ys), max(ys)
    for name, (k, c) in lines.items():
        out[name] = (c + k * y0, c + k * y1, float(np.degrees(np.arctan(k))))
    (l0, l1, _), (r0, r1, _) = out["left"], out["right"]
    out["keystone"] = float((r1 - l1) / max(r0 - l0, 1e-9) - 1)
    out.update(lines=lines, y=(float(y0), float(y1)), n=n)
    return out


def _vertical_support(pts, staves):
    """Conservative staff-crossing evidence, independent of stroke angle.

    A full-height stroke gets weight 1, a fragment gets at least 0.5.
    This is a geometric heuristic, not a calibrated barline classifier.
    """
    if not staves:
        return 1.0
    x = float(np.median(pts[:, 0]))
    ymin, ymax = np.min(pts[:, 1]), np.max(pts[:, 1])
    crossings = 0
    for st in staves:
        t = np.asarray(st.tuples)
        if not len(t):
            continue
        xs = t[:, 2, 0]
        if not xs.min() - st.spacing <= x <= xs.max() + st.spacing:
            continue
        ys = t[np.argmin(np.abs(xs - x)), :, 1]
        crossings = max(crossings, int(((ys >= ymin) & (ys <= ymax)).sum()))
    return 0.5 + 0.1 * crossings


def build_constraints(S: Structures, grid, priors=("staff", "jianpu", "text", "vertical"), tuple_u=0.3, vseg_w=0.6,
                      spacing_gate=0.2, ends_w=0.0, ends_mode="chain", vseg_balance=False):
    """``tuple_u`` ties U of the 5 staff lines at one *current* image column, which
    preserves whatever shear the current image has; 0 lets vertical strokes decide.
    ``spacing_gate``: a staff (jianpu row) whose spacing (digit height) deviates
    from the page median by more than this fraction keeps its own value, as a
    genuinely smaller staff would; larger values pull badly fitted staves back.
    ``ends_w``: the staves' left ends (clefs) and right ends share one U each
    (justified systems; an indented first system is an outlier for the robust
    solve), as in the margin constraints of Stamatopoulos et al. 2011.
    ``ends_mode``: "chain" ties consecutive staves' extreme tuples (a staff split
    into pieces, or an indented / short staff, breaks two links); "rows" one
    end per staff row with outliers removed (``_row_ends``)."""
    U, V = _System(grid), _System(grid)
    staves = S.staves if "staff" in priors else []
    s_all = [st.spacing for st in staves]
    s_ref = float(np.median(s_all)) if s_all else None
    for st in staves:
        s_t = st.spacing if (s_ref is None or abs(st.spacing - s_ref) > spacing_gate * s_ref) else s_ref
        T = np.array(st.tuples)  # (n,5,2)
        if len(T) == 0:
            continue
        loc = np.diff(T[:, :, 1], axis=1).mean(1)          # local spacing per tuple
        for i in range(4):
            V.diff(T[:, i + 1], T[:, i], s_t, 1.0)
            if tuple_u > 0:
                U.diff(T[:, i + 1], T[:, i], 0.0, tuple_u)
        xs_t = T[:, 0, 0]
        order = np.argsort(xs_t)
        for fi in st.frag_ids:
            p = _sub(S.fragments[fi].pts, 20)
            if len(p) < 2:
                continue
            V.diff(p[1:], p[:-1], 0.0, 1.0)
            sc = s_t / np.interp((p[1:, 0] + p[:-1, 0]) / 2, xs_t[order], loc[order])
            U.diff(p[1:], p[:-1], np.linalg.norm(np.diff(p, axis=0), axis=1) * sc, 0.5)
    if ends_w > 0 and len(staves) >= 3 and ends_mode == "rows":
        for side in (0, -1):
            pts = _row_ends(S, side)
            if len(pts) >= 3:
                U.diff(pts[1:], pts[:-1], 0.0, ends_w)
    elif ends_w > 0 and len(staves) >= 3:
        for side in (0, -1):
            pts = []
            for st in staves:
                T = np.array(st.tuples)
                if len(T):
                    k = np.argmin(T[:, 2, 0]) if side == 0 else np.argmax(T[:, 2, 0])
                    pts.append(T[k, 2])
            if len(pts) >= 3:
                pts = np.array(pts)
                U.diff(pts[1:], pts[:-1], 0.0, ends_w)
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
        h_t = h_ref if (r.kind == "jianpu" and h_ref and abs(r.height - h_ref) < spacing_gate * h_ref) else r.height
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
                w = vseg_w
                if vseg_balance and S.staves:
                    # Bound the squared weight per stroke to four intervals;
                    # leave jianpu-only pages unchanged in this experiment.
                    w *= _vertical_support(v.pts, S.staves) * np.sqrt(min(1., 4. / (len(p) - 1)))
                U.diff(p[1:], p[:-1], 0.0, w)
    return U, V



def _content_points(S: Structures):
    """All detected structure points (staff / free lines, glyph boxes, strokes)."""
    P = [f.pts for f in S.fragments] + [v.pts for v in S.vsegs]
    for r in S.rows:
        P.append(np.array([[b.x0, b.y0] for b in r.blobs] + [[b.x1, b.y1] for b in r.blobs], float))
    P = [np.asarray(p, float).reshape(-1, 2) for p in P if len(p)]
    return np.concatenate(P) if P else None


def _guard_field(UV, identity, node, min_scale=0.2, max_scale=5.0):
    """Backtrack until every bilinear cell has a well-conditioned Jacobian.

    Check all four corners, not only the centre: the determinant of a
    bilinear cell is affine, so positive corner values exclude local folds
    throughout that cell. This is a local check, not global injectivity.
    """
    identity = np.asarray(identity, dtype=float)
    if not np.isfinite(UV).all():
        return identity.copy(), 0.0
    for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
        trial = identity + alpha * (UV - identity)
        dx = np.diff(trial, axis=1) / node
        dy = np.diff(trial, axis=0) / node
        good = True
        for x in (dx[:-1], dx[1:]):
            for y in (dy[:, :-1], dy[:, 1:]):
                J = np.stack([x, y], axis=-1)
                sv = np.linalg.svd(J, compute_uv=False)
                if (np.any(np.linalg.det(J) <= 0) or np.any(sv[..., -1] < min_scale)
                        or np.any(sv[..., 0] > max_scale)):
                    good = False
                    break
            if not good:
                break
        if good:
            return trial, alpha
    return identity.copy(), 0.0


def fit_field(S: Structures, node=48, lam=6.0, mu=0.02, priors=("staff", "jianpu", "text", "vertical"),
              tuple_u=0.3, vseg_w=0.6, lam_u=None, lam_v=None, spacing_gate=0.2, ends_w=0.0, keep_scale=False,
              anchor=False, keep_pts=None, ends_mode="chain", guard=False, vseg_balance=False):
    """``lam_u`` / ``lam_v`` override the smoothness per component: V is pinned by
    many reliable staff lines, U only by short, noisier vertical strokes.
    ``keep_scale``: constrain U and V to unit mean gradient in the solve (global
    scale and aspect stay as the metric surface model set them).
    ``anchor``: place the result so that the detected content keeps its left /
    top margins, and grow the canvas instead of cutting content off.
    ``keep_pts``: further content points for the anchor (outlines of all
    printed content: notes below the staves, dynamics, exercise numbers),
    which the detected structures alone miss.
    ``guard``: backtrack displacement to keep each cell locally invertible
    with corner singular values between 0.2 and 5; fall back to identity.
    The returned map carries ``forward(pts)``: input -> output coordinates."""
    H, W = S.shape
    g = _Grid(W, H, node)
    U, V = build_constraints(S, g, priors, tuple_u=tuple_u, vseg_w=vseg_w, spacing_gate=spacing_gate, ends_w=ends_w,
                             ends_mode=ends_mode, vseg_balance=vseg_balance)
    if not U.A or not V.A:
        return None
    if keep_scale:
        # Gauge: "equal U along a stroke" / "equal V along a line" residuals
        # shrink when the whole component is compressed, so vertical weights
        # squeeze the page sideways (jianpu: up to 2x). The global scale is
        # already metric after B; the field may only fix local errors.
        U.mean_gradient(1, node, 1e3)
        V.mean_gradient(0, node, 1e3)
    gx, gy = np.meshgrid(np.arange(g.nx) * node, np.arange(g.ny) * node)
    sm = g.smooth_ops()
    u = U.solve(sm, lam if lam_u is None else lam_u, gx.ravel().astype(float), mu)
    v = V.solve(sm, lam if lam_v is None else lam_v, gy.ravel().astype(float), mu)
    UV = np.stack([u, v], 1).reshape(g.ny, g.nx, 2)
    alpha = 1.0
    if guard:
        UV, alpha = _guard_field(UV, np.stack([gx, gy], -1), node)
    # Report the accepted map's scale, including any backtracking.
    scale = tuple(float(np.diff(UV[..., k], axis=ax).mean() / node) for k, ax in ((0, 1), (1, 0)))
    pts = _content_points(S) if anchor else None
    if pts is not None and keep_pts is not None and len(keep_pts):
        pts = np.concatenate([pts, np.asarray(keep_pts, float).reshape(-1, 2)])
    if pts is not None and len(pts) >= 10:
        # keep the content's left / top margins and grow the canvas when the
        # content got wider or taller: a mean-displacement recentring lets the
        # deformed content slide off the canvas (clefs, exercise numbers)
        dst = g.weights(pts) @ UV.reshape(-1, 2)
        cmin, cmax = pts.min(0), pts.max(0)
        UV -= dst.min(0) - cmin
        dmax = dst.max(0) - (dst.min(0) - cmin)
        Wo = int(max(W, np.ceil(dmax[0] + (W - cmax[0]))))
        Ho = int(max(H, np.ceil(dmax[1] + (H - cmax[1]))))
    else:
        # recentre so that the content stays on the same canvas
        UV -= (UV - np.stack([gx, gy], -1)).reshape(-1, 2).mean(0)
        Wo, Ho = W, H

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

    bm = BackwardMap.from_function(fn, (Wo, Ho), stride=4)
    bm.forward_nodes = UV
    bm.forward = lambda p: g.weights(p) @ UV.reshape(-1, 2)
    bm.field_scale = scale
    bm.field_step = alpha
    return bm
