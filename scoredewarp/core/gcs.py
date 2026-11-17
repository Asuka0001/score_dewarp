"""Stage B (metric): pinhole camera + general cylindrical surface (GCS).

Paper bends without stretching, so the page is modelled as a developable
general cylinder (Meng et al., "Metric rectification of curved document
images", TPAMI 2012; also the model behind page_dewarp and smude):

  rulings run in direction  b = (-sin phi, cos phi)  of the flat page,
  across them the cross-section is the arc-length curve
      x'(s) = int_0^s cos theta,   z(s) = int_0^s sin theta,
  with the slope profile theta(s) piecewise linear on K knots.

A flat page point (X, Y) maps isometrically to 3-D, so the flattened output is
exact -- no arc-length approximation is needed afterwards, and the horizontal
scale under curl and perspective comes from the model rather than from a guess.

Unknowns: camera rotation/translation (6), ruling angle phi (1), slope knots (K),
and per structure its flat coordinates (as in the legacy surface model):
  staff:   Y = Y0 + line * s      (5 equidistant lines; spacing shared softly)
  jianpu/text row: top Y = T, bottom Y = T + h  (digit double line)
  free horizontal fragment: Y = const
  vertical segment (barline/stem): X = const
  staff left / right ends: X = X_left / X_right shared by all staves (justified
           systems; an indented first system is handled by the robust loss)
Every point on a horizontal structure has its own free X.
Residuals: reprojection error (px, soft-L1) plus smoothness / gauge priors.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .structures import Structures
from .transforms import BackwardMap

K_KNOTS = 13
N_GLOBAL = 6 + 1 + K_KNOTS
TABLE = 1201


def _subsample(pts, min_d):
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= min_d:
            keep.append(i)
    if keep[-1] != len(pts) - 1 and len(pts) > 1:
        keep.append(len(pts) - 1)
    return pts[keep]


class GCSModel:
    def __init__(self, img_size, focal=None):
        W0, H0 = img_size
        self.img_size = (W0, H0)
        self.f = focal or 1.0 * max(W0, H0)
        self.c = np.array([W0 / 2, H0 / 2])
        self.K = np.array([[self.f, 0, self.c[0]], [0, self.f, self.c[1]], [0, 0, 1]])

    # ------------------------------------------------------------ build --
    def build(self, S: Structures, H_coarse: np.ndarray, weights=None):
        w_type = {"staff": 1.0, "hline": 0.5, "jianpu": 0.8, "text": 0.5, "vseg": 0.6, "ends": 0.5}
        w_type.update(weights or {})
        Hinv = np.linalg.inv(H_coarse)
        W1 = S.shape[1]
        self.unit = float(W1)
        P = [0.0] * N_GLOBAL
        obs, xa, ya, yb, cyb, wts, content = [], [], [], [], [], [], []

        def new(v):
            P.append(float(v))
            return len(P) - 1

        def add(pt, ix, iy, iyb=0, cy=0.0, w=1.0, is_content=True):
            obs.append(pt), xa.append(ix), ya.append(iy), yb.append(iyb), cyb.append(cy)
            wts.append(w), content.append(is_content)

        sp_unit = max(S.spacing_est, 4.0)
        staff_s, ends = [], {"L": [], "R": []}
        for st in S.staves:
            y_top = np.median([t[0, 1] for t in st.tuples])
            iY, iS = new(y_top / W1), new(st.spacing / W1)
            staff_s.append(iS)
            for fi in st.frag_ids:
                f = S.fragments[fi]
                for p in _subsample(f.pts, 2.0 * st.spacing):
                    add(p, new(p[0] / W1), iY, iS, float(f.line), w_type["staff"])
            # end points of the system (outer lines only, where they were traced)
            for fi in st.frag_ids:
                f = S.fragments[fi]
                if f.line in (0, 4):
                    ends["L"].append((f.pts[0], iY, iS, float(f.line)))
                    ends["R"].append((f.pts[-1], iY, iS, float(f.line)))
        self.staff_s = staff_s
        if len(S.staves) >= 3:
            for side in ("L", "R"):
                xs = [e[0][0] for e in ends[side]]
                iX = new(np.median(xs) / W1)
                for p, iY, iS, line in ends[side]:
                    add(p, iX, iY, iS, line, w_type["ends"], False)
        for f in S.fragments:
            if f.staff >= 0:
                continue
            L = np.linalg.norm(f.pts[-1] - f.pts[0])
            if L < 2.5 * sp_unit or L < 40:
                continue
            iY = new(np.mean(f.pts[:, 1]) / W1)
            for p in _subsample(f.pts, 40):
                add(p, new(p[0] / W1), iY, w=w_type["hline"], is_content=False)
        jp_h = []
        for r in S.rows:
            if len(r.full) < 3:
                continue
            fb = [r.blobs[i] for i in r.full]
            iT = new(np.median([b.y0 for b in fb]) / W1)
            iH = new(r.height / W1)
            if r.kind == "jianpu":
                jp_h.append(iH)
            w = w_type["jianpu" if r.kind == "jianpu" else "text"]
            for b in fb:
                add(np.array([b.cx, b.y0]), new(b.cx / W1), iT, 0, 0.0, w)
                add(np.array([b.cx, b.y1]), new(b.cx / W1), iT, iH, 1.0, w)
        self.jp_h = jp_h
        for v in S.vsegs:
            ix = new(np.mean(v.pts[:, 0]) / W1)
            for p in _subsample(v.pts, max(15, 0.4 * (v.pts[-1, 1] - v.pts[0, 1]))):
                add(p, ix, new(p[1] / W1), w=w_type["vseg"], is_content=False)
        if not obs:
            return 0
        obs1 = np.array(obs, float)
        self.obs = cv2.perspectiveTransform(obs1.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        self.xa, self.ya, self.yb = np.array(xa), np.array(ya), np.array(yb)
        self.cyb, self.w = np.array(cyb), np.array(wts)
        self.content = np.array(content, bool)
        if self.content.sum() < 10:
            self.content[:] = True
        self.p0 = np.array(P)
        Xs, Ys = obs1[:, 0] / W1, obs1[:, 1] / W1
        self.Xc, self.Yc = 0.5 * (Xs.min() + Xs.max()), 0.5 * (Ys.min() + Ys.max())
        # Slope knots span the observed structures only (any ruling direction);
        # beyond them np.interp holds theta constant, i.e. the sheet continues
        # along its tangent instead of folding where nothing constrains it.
        Hc_, Wc_ = S.shape
        rc = 1.02 * np.max(np.hypot(Xs - self.Xc, Ys - self.Yc))
        corners = np.array([[0, 0], [Wc_, 0], [0, Hc_], [Wc_, Hc_]], float) / W1
        self.R = max(rc, 1.6 * np.max(np.hypot(corners[:, 0] - self.Xc, corners[:, 1] - self.Yc)))
        self.knots = np.linspace(-rc, rc, K_KNOTS)
        self.sg = np.linspace(-self.R, self.R, TABLE)
        self.mid = TABLE // 2
        # camera init: page plane = coarse image plane, via PnP
        gx, gy = np.meshgrid(np.linspace(0, Wc_, 6), np.linspace(0, Hc_, 8))
        g1 = np.stack([gx.ravel(), gy.ravel()], 1)
        g0 = cv2.perspectiveTransform(g1.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        objp = np.concatenate([g1 / W1, np.zeros((len(g1), 1))], 1)
        ok, rvec, tvec = cv2.solvePnP(objp, g0, self.K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        self.p0[0:3], self.p0[3:6] = rvec.ravel(), tvec.ravel()
        self.tz0 = float(tvec.ravel()[2])
        return len(obs)

    # ------------------------------------------------------------ model --
    def _profile(self, p):
        th = np.interp(self.sg, self.knots, p[7:7 + K_KNOTS])
        ds = self.sg[1] - self.sg[0]
        cx = np.concatenate([[0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * ds)])
        cz = np.concatenate([[0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * ds)])
        return cx - cx[self.mid], cz - cz[self.mid]

    def surface(self, p, X, Y):
        """Flat page (X, Y) -> 3-D point on the general cylinder."""
        phi = p[6]
        c, s = np.cos(phi), np.sin(phi)
        dX, dY = X - self.Xc, Y - self.Yc
        u = c * dX + s * dY
        r = -s * dX + c * dY
        cx, cz = self._profile(p)
        xp = np.interp(u, self.sg, cx)
        z = np.interp(u, self.sg, cz)
        # linear extrapolation beyond the table (rare: points far outside the canvas)
        lo, hi = u < self.sg[0], u > self.sg[-1]
        if lo.any() or hi.any():
            th = p[7:7 + K_KNOTS]
            xp = np.where(lo, cx[0] + (u - self.sg[0]) * np.cos(th[0]), xp)
            z = np.where(lo, cz[0] + (u - self.sg[0]) * np.sin(th[0]), z)
            xp = np.where(hi, cx[-1] + (u - self.sg[-1]) * np.cos(th[-1]), xp)
            z = np.where(hi, cz[-1] + (u - self.sg[-1]) * np.sin(th[-1]), z)
        return np.stack([self.Xc + c * xp - s * r, self.Yc + s * xp + c * r, z], -1)

    def project(self, p, X, Y):
        R, _ = cv2.Rodrigues(p[0:3])
        Pc = self.surface(p, X, Y) @ R.T + p[3:6]
        zc = Pc[:, 2:3]
        out = self.f * Pc[:, :2] / np.where(zc > 1e-6, zc, np.nan) + self.c
        return out

    def _XY(self, p):
        return p[self.xa], p[self.ya] + self.cyb * p[self.yb]

    def _priors(self, p):
        th = p[7:7 + K_KNOTS]
        pri = [np.diff(th, 2) * self.lam_curv, th * self.lam_th, [th[K_KNOTS // 2] * 200.0]]
        u = self.unit
        for grp, lam in ((self.staff_s, self.lam_sp), (self.jp_h, self.lam_h)):
            if len(grp) > 1:
                v = p[grp]
                pri.append((v - np.median(v)) * u * lam)
        pri.append([(p[5] - self.tz0) / self.tz0 * 20.0])
        return np.concatenate([np.atleast_1d(np.asarray(x, float)) for x in pri])

    def residuals(self, p):
        X, Y = self._XY(p)
        uv = self.project(p, X, Y)
        r = (uv - self.obs) * self.w[:, None]
        r = np.where(np.isfinite(r), r, 1e4)
        return np.concatenate([r.ravel(), self._priors(p)])

    def sparsity(self):
        n_obs, n_p = len(self.obs), len(self.p0)
        n_pri = len(self._priors(self.p0))
        A = lil_matrix((2 * n_obs + n_pri, n_p), dtype=np.uint8)
        rows = np.arange(n_obs)
        for d in (0, 1):
            rr = 2 * rows + d
            A[rr[:, None], np.arange(N_GLOBAL)[None, :]] = 1
            A[rr, self.xa] = 1
            A[rr, self.ya] = 1
            m = self.cyb != 0
            A[rr[m], self.yb[m]] = 1
        # prior rows: few and cheap, so mark every parameter they can touch
        k = 2 * n_obs
        cols = list(range(7, 7 + K_KNOTS)) + [5] + list(self.staff_s) + list(self.jp_h)
        A[k:k + n_pri, cols] = 1
        return A

    def fit(self, lam_curv=40.0, lam_th=0.3, lam_sp=2.0, lam_h=1.0, max_nfev=200, phis=(0.0, np.pi / 2),
            warm=False):
        """``warm``: continue from ``self.p0`` as is (no ruling-direction restarts)."""
        self.lam_curv, self.lam_th, self.lam_sp, self.lam_h = lam_curv, lam_th, lam_sp, lam_h
        self.fit_kw = dict(lam_curv=lam_curv, lam_th=lam_th, lam_sp=lam_sp, lam_h=lam_h, max_nfev=max_nfev)
        n = 2 * len(self.obs)
        r0 = self.residuals(self.p0)
        self.rms0 = float(np.sqrt(np.mean(r0[:n] ** 2)))
        best = None
        sp = self.sparsity()
        kw = dict(jac_sparsity=sp, method="trf", loss="soft_l1", f_scale=3.0, x_scale="jac")
        if warm:
            best = least_squares(self.residuals, self.p0, max_nfev=max_nfev, **kw)
        else:
            for phi in phis:  # short runs per ruling direction, then finish the best
                p0 = self.p0.copy()
                p0[6] = phi
                res = least_squares(self.residuals, p0, max_nfev=max(20, max_nfev // 5), **kw)
                if best is None or res.cost < best.cost:
                    best = res
            best = least_squares(self.residuals, best.x, max_nfev=max_nfev, **kw)
        self.p = best.x
        self.rms = float(np.sqrt(np.mean(best.fun[:n] ** 2)))
        e = np.hypot(*best.fun[:n].reshape(-1, 2).T) / np.maximum(self.w, 1e-9)
        self.err_median = float(np.median(e))
        self.err_p90 = float(np.percentile(e, 90))
        self.cost = float(best.cost)
        return best

    # ------------------------------------------------------------- edges --
    def flat_coords(self, q, iters=6):
        """Inverse projection: image points (N, 2) -> flat page (X, Y) on the fitted surface."""
        p = self.p
        x0, x1, y0, y1 = self.content_extent(0.0)
        gx, gy = (x1 - x0), (y1 - y0)
        Xg, Yg = np.meshgrid(np.linspace(x0 - gx, x1 + gx, 120), np.linspace(y0 - gy, y1 + gy, 150))
        Xg, Yg = Xg.ravel(), Yg.ravel()
        qg = self.project(p, Xg, Yg)
        ok = np.isfinite(qg).all(1)
        from scipy.spatial import cKDTree
        _, j = cKDTree(qg[ok]).query(q)
        X, Y = Xg[ok][j].copy(), Yg[ok][j].copy()
        h = 1e-4
        for _ in range(iters):  # Gauss-Newton on the 2x2 local Jacobian
            f0 = self.project(p, X, Y)
            fx = (self.project(p, X + h, Y) - f0) / h
            fy = (self.project(p, X, Y + h) - f0) / h
            r = q - f0
            det = fx[:, 0] * fy[:, 1] - fx[:, 1] * fy[:, 0]
            det = np.where(np.abs(det) > 1e-12, det, np.nan)
            dX = (fy[:, 1] * r[:, 0] - fy[:, 0] * r[:, 1]) / det
            dY = (-fx[:, 1] * r[:, 0] + fx[:, 0] * r[:, 1]) / det
            X, Y = X + np.nan_to_num(dX), Y + np.nan_to_num(dY)
        bad = ~np.isfinite(self.project(p, X, Y)).all(1)
        X[bad], Y[bad] = np.nan, np.nan
        return X, Y

    @staticmethod
    def _ransac_line(a, b, tol, iters=200, seed=0):
        """Inliers of b = c0 + c1 * a (an edge is straight in the flat page)."""
        rng = np.random.default_rng(seed)
        n = len(a)
        best = np.zeros(n, bool)
        for _ in range(iters):
            i, j = rng.choice(n, 2, replace=False)
            if abs(a[j] - a[i]) < 1e-9:
                continue
            c1 = (b[j] - b[i]) / (a[j] - a[i])
            if abs(c1) > 0.2:   # an edge is never far from its axis after A
                continue
            inl = np.abs(b - (b[i] + c1 * (a - a[i]))) < tol
            if inl.sum() > best.sum():
                best = inl
        return best

    def select_edges(self, edges_coarse, H_coarse, tol=0.006, min_span=0.25):
        """Keep the edge points that are straight in the flat page (RANSAC).

        The paper outline from a brightness mask follows cast shadows and
        occluders; those stretches are curved in the flat page while the real
        border is a straight line X or Y = const there. Needs a fitted model.
        """
        Hinv = np.linalg.inv(H_coarse)
        x0, x1, y0, y1 = self.content_extent(0.0)
        size = max(x1 - x0, y1 - y0)
        OX, OY = self._XY(self.p)   # content structures lie on the sheet
        OX, OY = OX[self.content], OY[self.content]
        out, info = {}, {}
        self.edge_flat = {}         # side -> flat X or Y of the border (current surface)
        for side, pts in edges_coarse.items():
            q = cv2.perspectiveTransform(np.asarray(pts, float).reshape(-1, 1, 2), Hinv).reshape(-1, 2)
            X, Y = self.flat_coords(q)
            ok = np.isfinite(X)
            if ok.sum() < 10:
                continue
            a, b = (X, Y) if side in ("top", "bottom") else (Y, X)
            inl = np.zeros(len(q), bool)
            inl[ok] = self._ransac_line(a[ok], b[ok], tol * size)
            span = np.ptp(a[inl]) if inl.sum() > 1 else 0.0
            length = (x1 - x0) if side in ("top", "bottom") else (y1 - y0)
            info[side] = (int(inl.sum()), len(q), float(span / max(length, 1e-9)))
            if inl.sum() < 8 or span < min_span * length:
                continue
            # a border with structures beyond it is a shadow / light edge, not the sheet
            e = np.median(b[inl])
            beyond = {"top": OY < e - tol * size, "bottom": OY > e + tol * size,
                      "left": OX < e - tol * size, "right": OX > e + tol * size}[side]
            info[side] += (float(beyond.mean()),)
            if beyond.mean() > 0.005:
                continue
            out[side] = np.asarray(pts, float)[inl]
            self.edge_flat[side] = float(e)
        self.edge_info = info
        return out

    def add_edges(self, edges_coarse, H_coarse, w=1.0, widen=True):
        """Append page-border observations (top/bottom: Y = const, left/right: X = const)
        to a fitted model and prepare a warm restart from the current solution."""
        Hinv = np.linalg.inv(H_coarse)
        P = list(self.p)
        n_old = len(P)
        obs, xa, ya = [], [], []
        self.edge_par, self.edge_rows = {}, {}
        for side, pts in edges_coarse.items():
            q = cv2.perspectiveTransform(np.asarray(pts, float).reshape(-1, 1, 2), Hinv).reshape(-1, 2)
            X, Y = self.flat_coords(q)
            ok = np.isfinite(X)
            q, X, Y = q[ok], X[ok], Y[ok]
            if len(q) < 8:
                continue
            horiz = side in ("top", "bottom")
            P.append(float(np.median(Y if horiz else X)))
            ie = len(P) - 1
            self.edge_par[side] = ie
            r0 = len(self.obs) + len(obs)
            for qi, Xi, Yi in zip(q, X, Y):
                P.append(float(Xi if horiz else Yi))
                obs.append(qi)
                xa.append(len(P) - 1 if horiz else ie)
                ya.append(ie if horiz else len(P) - 1)
            self.edge_rows[side] = np.arange(r0, len(self.obs) + len(obs))
        if not obs:
            return 0
        k = len(obs)
        self.obs = np.concatenate([self.obs, np.array(obs)])
        self.xa = np.concatenate([self.xa, xa])
        self.ya = np.concatenate([self.ya, ya])
        self.yb = np.concatenate([self.yb, np.zeros(k, int)])
        self.cyb = np.concatenate([self.cyb, np.zeros(k)])
        self.w = np.concatenate([self.w, np.full(k, w)])
        self.content = np.concatenate([self.content, np.zeros(k, bool)])
        # widen the slope knots to the borders (theta is constant beyond the old ones)
        X, Y = self._XY(np.array(P))
        rc = 1.02 * np.max(np.hypot(X - self.Xc, Y - self.Yc))
        if widen and rc > self.knots[-1]:
            new = np.linspace(-rc, rc, K_KNOTS)
            P[7:7 + K_KNOTS] = list(np.interp(new, self.knots, self.p[7:7 + K_KNOTS]))
            self.knots = new
            self.R = max(self.R, 1.2 * rc)
            self.sg = np.linspace(-self.R, self.R, TABLE)
        self.p0 = np.array(P)
        return k

    def edge_errors(self):
        """Median reprojection error (px) per page side after the fit."""
        X, Y = self._XY(self.p)
        e = np.hypot(*(self.project(self.p, X, Y) - self.obs).T)
        return {side: float(np.nanmedian(e[r])) for side, r in getattr(self, "edge_rows", {}).items()}

    # ------------------------------------------------------------ output --
    def content_extent(self, margin=0.05):
        X, Y = self._XY(self.p)
        X, Y = X[self.content], Y[self.content]
        x0, x1 = np.percentile(X, [0.5, 99.5])
        y0, y1 = np.percentile(Y, [0.5, 99.5])
        m = margin * (x1 - x0)
        return x0 - m, x1 + m, y0 - m, y1 + m

    def page_extent(self, page_mask=None, grow=0.35, margin=0.01, edges=(), edge_margin=0.015):
        """Flat box of all page points visible in the photo (inside the page
        mask when given), limited to the content box grown by ``grow``.

        Sides listed in ``edges`` (fitted paper borders, see ``add_edges``) are
        taken from the border itself instead, without the ``grow`` limit."""
        box = self._visible_extent(page_mask, grow, margin)
        if not edges:
            return box
        x0, x1, y0, y1 = box
        m = edge_margin * (x1 - x0)
        par = getattr(self, "edge_par", {})
        val = {side: (self.p[par[side]] if side in par else self.edge_flat[side]) for side in edges}
        x0 = val["left"] - m if "left" in val else x0
        x1 = val["right"] + m if "right" in val else x1
        y0 = val["top"] - m if "top" in val else y0
        y1 = val["bottom"] + m if "bottom" in val else y1
        if x1 <= x0 or y1 <= y0:
            return box
        return x0, x1, y0, y1

    def _visible_extent(self, page_mask, grow, margin):
        x0, x1, y0, y1 = self.content_extent(0.0)
        gx, gy = (x1 - x0) * grow, (y1 - y0) * grow
        bx0, bx1, by0, by1 = x0 - gx, x1 + gx, y0 - gy, y1 + gy
        Xg, Yg = np.meshgrid(np.linspace(bx0, bx1, 160), np.linspace(by0, by1, 200))
        q = self.project(self.p, Xg.ravel(), Yg.ravel())
        W0, H0 = self.img_size
        ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < W0 - 1) & (q[:, 1] >= 0) & (q[:, 1] < H0 - 1)
        if page_mask is not None and ok.any():
            qi = np.round(q[ok]).astype(int)
            inside = page_mask[qi[:, 1], qi[:, 0]] > 0
            ok2 = ok.copy()
            ok2[ok] = inside
            if ok2.sum() > 0.3 * ok.sum():
                ok = ok2
        if ok.sum() < 50:
            return self.content_extent()
        X, Y = Xg.ravel()[ok], Yg.ravel()[ok]
        ex = (min(np.percentile(X, 0.2), x0), max(np.percentile(X, 99.8), x1),
              min(np.percentile(Y, 0.2), y0), max(np.percentile(Y, 99.8), y1))
        m = margin * (ex[1] - ex[0])
        return ex[0] - m, ex[1] + m, ex[2] - m, ex[3] + m

    def backward_map(self, extent, out_scale=1.0, stride=4):
        x0, x1, y0, y1 = extent
        ppu = self.unit * out_scale
        W, H = int(np.ceil((x1 - x0) * ppu)), int(np.ceil((y1 - y0) * ppu))
        p = self.p

        def fn(q):
            return self.project(p, x0 + q[:, 0] / ppu, y0 + q[:, 1] / ppu)

        self.out_to_page = lambda q: np.stack([x0 + q[:, 0] / ppu, y0 + q[:, 1] / ppu], 1)
        return BackwardMap.from_function(fn, (W, H), stride)
