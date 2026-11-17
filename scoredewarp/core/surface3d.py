"""Stage B: parametric 3-D page surface + pinhole camera fitted to music structure.

Unknowns
  global : camera rotation r (3), translation t (3), surface coefficients c (8)
  per horizontal structure : its flat Y (staff: Y0 + line*spacing, shared spacing)
  per point on a horizontal structure : its flat X
  per glyph : flat X of its top and bottom point (verticality comes from strokes/barlines)
  per text row : Y_top and glyph height h                       (baseline/topline prior)
  per vertical segment : one flat X, per point its flat Y       (barline/stem prior)

The surface is z(X,Y) = s * sum_m c_m b_m(xi, eta) with low-order monomials, so
it represents curl, bends and gentle non-developable bumps; perspective is
handled by the camera. Residuals are reprojection errors (pixels) with a robust
loss, plus score-type priors:
  * 五线谱: 5 lines equally spaced, all staves share (nearly) one spacing
  * 简谱  : note-row digits share baseline+topline, uniform digit height across
           jianpu rows; lyrics rows same (own height)
  * both  : barlines / stems vertical.
The output map is reparameterised by surface arc length so curled regions are
not compressed.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .structures import Structures
from .transforms import BackwardMap

N_C = 8


def _basis(xi, eta):
    return np.stack([xi ** 2, xi ** 3, xi ** 4, eta ** 2, eta ** 3, xi * eta, xi ** 2 * eta, xi * eta ** 2], -1)


def _basis_dxi(xi, eta):
    z = np.zeros_like(xi)
    return np.stack([2 * xi, 3 * xi ** 2, 4 * xi ** 3, z, z, eta, 2 * xi * eta, eta ** 2], -1)


def _basis_deta(xi, eta):
    z = np.zeros_like(xi)
    return np.stack([z, z, z, 2 * eta, 3 * eta ** 2, xi, xi ** 2, 2 * xi * eta], -1)


def _subsample(pts, min_d):
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= min_d:
            keep.append(i)
    if keep[-1] != len(pts) - 1 and len(pts) > 1:
        keep.append(len(pts) - 1)
    return pts[keep]


class SurfaceModel:
    def __init__(self, img_size, focal=None):
        W0, H0 = img_size
        self.f = focal or 1.0 * max(W0, H0)
        self.c = np.array([W0 / 2, H0 / 2])
        self.K = np.array([[self.f, 0, self.c[0]], [0, self.f, self.c[1]], [0, 0, 1]])

    # ------------------------------------------------------------------
    def build(self, S: Structures, H_coarse: np.ndarray, w_type=None):
        """Collect observations from structures detected in the coarse image."""
        w_type = w_type or {}
        Hinv = np.linalg.inv(H_coarse)
        W1 = S.shape[1]
        self.unit = float(W1)  # coarse px per page unit
        P = []  # param initial values
        obs, xa, ya, yb, cyb, wts, content = [], [], [], [], [], [], []
        prior_rows = []  # (kind, idx list)

        def new(v):
            P.append(float(v))
            return len(P) - 1

        self.n_global = 6 + N_C
        P.extend([0.0] * self.n_global)

        def add_obs(pt1, ix, iy, iyb=0, cy=0.0, w=1.0, is_content=True):
            content.append(is_content)
            obs.append(pt1)
            xa.append(ix)
            ya.append(iy)
            yb.append(iyb)
            cyb.append(cy)
            wts.append(w)

        sp_unit = max(S.spacing_est, 4.0)
        # --- staves (五线谱 prior) ---
        staff_s = []
        for k, st in enumerate(S.staves):
            y_top = np.median([t[0, 1] for t in st.tuples])
            iY = new(y_top / W1)
            iS = new(st.spacing / W1)
            staff_s.append(iS)
            for fi in st.frag_ids:
                f = S.fragments[fi]
                for p in _subsample(f.pts, 2.0 * st.spacing):
                    add_obs(p, new(p[0] / W1), iY, iS, float(f.line), w_type.get("staff", 1.0))
        self.staff_s = staff_s
        # --- free horizontal fragments (underlines, ledger lines, ...) ---
        for f in S.fragments:
            if f.staff >= 0:
                continue
            L = np.linalg.norm(f.pts[-1] - f.pts[0])
            if L < 2.5 * sp_unit or L < 40:
                continue
            iY = new(np.mean(f.pts[:, 1]) / W1)
            for p in _subsample(f.pts, 40):
                add_obs(p, new(p[0] / W1), iY, w=w_type.get("hline", 0.5), is_content=False)
        # --- text rows (简谱 / lyrics prior) ---
        jp_h = []
        for r in S.rows:
            if len(r.full) < 3:
                continue
            fb = [r.blobs[i] for i in r.full]
            top = np.median([b.y0 for b in fb])
            iT = new(top / W1)
            iH = new(r.height / W1)
            if r.kind == "jianpu":
                jp_h.append(iH)
            w = w_type.get("jianpu" if r.kind == "jianpu" else "text", 0.8 if r.kind == "jianpu" else 0.5)
            for b in fb:
                # top / bottom get independent X: a bbox centre carries no shear
                # information, so tying them would freeze the photo's shear in.
                add_obs(np.array([b.cx, b.y0]), new(b.cx / W1), iT, 0, 0.0, w)
                add_obs(np.array([b.cx, b.y1]), new(b.cx / W1), iT, iH, 1.0, w)
        self.jp_h = jp_h
        # --- vertical segments ---
        for v in S.vsegs:
            ix = new(np.mean(v.pts[:, 0]) / W1)
            for p in _subsample(v.pts, max(15, 0.4 * (v.pts[-1, 1] - v.pts[0, 1]))):
                add_obs(p, ix, new(p[1] / W1), w=w_type.get("vseg", 0.6), is_content=False)

        # No detected evidence is a normal case for generic document photos.
        # Let the pipeline keep its coarse correction instead of passing an
        # empty array to OpenCV (which returns None and then fails on reshape).
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
        # basis normalisation
        Xs, Ys = obs1[:, 0] / W1, obs1[:, 1] / W1
        self.Xc, self.Yc = 0.5 * (Xs.min() + Xs.max()), 0.5 * (Ys.min() + Ys.max())
        self.hx = max(0.5 * (Xs.max() - Xs.min()), 0.1)
        self.hy = max(0.5 * (Ys.max() - Ys.min()), 0.1)
        self.meanX0, self.meanY0 = Xs.mean(), Ys.mean()
        # camera init: page plane = coarse image plane (z=0), via PnP on a grid
        gx, gy = np.meshgrid(np.linspace(0, S.shape[1], 6), np.linspace(0, S.shape[0], 8))
        g1 = np.stack([gx.ravel(), gy.ravel()], 1)
        g0 = cv2.perspectiveTransform(g1.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        objp = np.concatenate([g1 / W1, np.zeros((len(g1), 1))], 1)
        ok, rvec, tvec = cv2.solvePnP(objp, g0, self.K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        self.p0[0:3] = rvec.ravel()
        self.p0[3:6] = tvec.ravel()
        self.tz0 = float(tvec.ravel()[2])
        return len(obs)

    # ------------------------------------------------------------------
    def surface_z(self, X, Y, c):
        xi, eta = (X - self.Xc) / self.hx, (Y - self.Yc) / self.hy
        return self.hx * (_basis(xi, eta) @ c)

    def project(self, p, X, Y):
        r, t, c = p[0:3], p[3:6], p[6:6 + N_C]
        R, _ = cv2.Rodrigues(r)
        P3 = np.stack([X, Y, self.surface_z(X, Y, c)], 1)
        Pc = P3 @ R.T + t
        return self.f * Pc[:, :2] / Pc[:, 2:3] + self.c

    def _XY(self, p):
        return p[self.xa], p[self.ya] + self.cyb * p[self.yb]

    def residuals(self, p):
        X, Y = self._XY(p)
        uv = self.project(p, X, Y)
        r = ((uv - self.obs) * self.w[:, None]).ravel()
        pri = [p[6:6 + N_C] * self.lam_c]
        u = self.unit
        if len(self.staff_s) > 1:
            s = p[self.staff_s]
            pri.append((s - s.mean()) * u * self.lam_sp)
        if len(self.jp_h) > 1:
            h = p[self.jp_h]
            pri.append((h - h.mean()) * u * self.lam_h)
        pri.append(np.array([(p[3 + 2] - self.tz0) / self.tz0 * 50.0]))
        return np.concatenate([r] + pri)

    def sparsity(self):
        n_obs = len(self.obs)
        n_p = len(self.p0)
        n_pri = N_C + (len(self.staff_s) if len(self.staff_s) > 1 else 0) + \
            (len(self.jp_h) if len(self.jp_h) > 1 else 0) + 1
        A = lil_matrix((2 * n_obs + n_pri, n_p), dtype=np.uint8)
        rows = np.arange(n_obs)
        for d in (0, 1):
            rr = 2 * rows + d
            A[rr[:, None], np.arange(self.n_global)[None, :]] = 1
            A[rr, self.xa] = 1
            A[rr, self.ya] = 1
            m = self.cyb != 0
            A[rr[m], self.yb[m]] = 1
        k = 2 * n_obs
        for i in range(N_C):
            A[k, 6 + i] = 1
            k += 1
        for grp in (self.staff_s, self.jp_h):
            if len(grp) > 1:
                for i in grp:
                    A[k, grp] = 1
                    k += 1
        A[k, 5] = 1
        return A

    def fit(self, lam_c=3.0, lam_sp=2.0, lam_h=1.0, max_nfev=200, verbose=0):
        self.lam_c, self.lam_sp, self.lam_h = lam_c, lam_sp, lam_h
        r0 = self.residuals(self.p0)
        res = least_squares(self.residuals, self.p0, jac_sparsity=self.sparsity(), method="trf",
                            loss="soft_l1", f_scale=3.0, x_scale="jac", max_nfev=max_nfev, verbose=verbose)
        self.p = res.x
        n = 2 * len(self.obs)
        self.rms0 = float(np.sqrt(np.mean(r0[:n] ** 2)))
        self.rms = float(np.sqrt(np.mean(res.fun[:n] ** 2)))
        return res

    # ------------------------------------------------------------------
    def image_extent(self, size):
        """Page-unit box covering a whole input image instead of detected content.

        Fits an affine map from observed image points to their fitted page
        coordinates and transforms the image corners. Intended for inputs that
        are already nearly rectified (e.g. a learned map's output), where the
        affine approximation is accurate and nothing outside detected rows
        should be cropped.
        """
        X, Y = self._XY(self.p)
        A = np.concatenate([self.obs, np.ones((len(self.obs), 1))], 1)
        coef, *_ = np.linalg.lstsq(A, np.stack([X, Y], 1), rcond=None)
        W, H = size
        corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], float) @ coef
        return (corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(), corners[:, 1].max())

    def backward_map(self, margin=0.05, stride=4, out_scale=1.0, extent=None):
        p = self.p
        if extent is not None:
            x0, x1, y0, y1 = extent
        else:
            X, Y = self._XY(p)
            # output extent from musical content (staves, rows), not page edges / clutter
            X, Y = X[self.content], Y[self.content]
            x0, x1 = np.percentile(X, [0.5, 99.5])
            y0, y1 = np.percentile(Y, [0.5, 99.5])
            mx = margin * (x1 - x0)
            x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - mx, y1 + mx
        # arc-length reparameterisation on a fine grid
        n = 300
        Xg, Yg = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
        c = p[6:6 + N_C]
        xi, eta = (Xg - self.Xc) / self.hx, (Yg - self.Yc) / self.hy
        dzdx = _basis_dxi(xi, eta) @ c            # hx * d/dX(...) / hx
        dzdy = (_basis_deta(xi, eta) @ c) * self.hx / self.hy
        dX = (x1 - x0) / (n - 1)
        dY = (y1 - y0) / (n - 1)
        Sx = np.concatenate([np.zeros((n, 1)), np.cumsum(np.sqrt(1 + dzdx ** 2)[:, :-1] * dX, 1)], 1)
        Ty = np.concatenate([np.zeros((1, n)), np.cumsum(np.sqrt(1 + dzdy ** 2)[:-1, :] * dY, 0)], 0)
        # use the central column / row profile (separable approximation keeps grid lines straight)
        Sx_row = Sx.mean(0)
        Ty_col = Ty.mean(1)
        ppu = self.unit * out_scale
        Wout = int(np.ceil(Sx_row[-1] * ppu))
        Hout = int(np.ceil(Ty_col[-1] * ppu))
        Xline, Yline = Xg[0], Yg[:, 0]

        def fn(q):
            s = q[:, 0] / ppu
            t = q[:, 1] / ppu
            Xq = np.interp(s, Sx_row, Xline)
            Yq = np.interp(t, Ty_col, Yline)
            return self.project(p, Xq, Yq)

        self.out_to_page = lambda q: np.stack([np.interp(q[:, 0] / ppu, Sx_row, Xline),
                                              np.interp(q[:, 1] / ppu, Ty_col, Yline)], 1)
        return BackwardMap.from_function(fn, (Wout, Hout), stride)
