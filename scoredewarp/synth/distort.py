"""Physically-motivated photo simulation of a flat score page.

Page model: developable curl along X (arc-length parameterised slope profile
theta(s)), optional mild bend along Y and small non-developable bumps, then a
random 3-D pose and a pinhole camera. On top: Lambertian shading, cast
shadows, uneven global illumination, finger occlusion, blur, noise, JPEG.

Returns the photo, the forward map F(flat_px) -> image_px (as a callable and
applied to all annotations) and a dense GT backward map image -> flat.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import CubicSpline, LinearNDInterpolator


def _rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class PageWarp:
    def __init__(self, page_wh, out_wh, rng, strength=1.0):
        Wf, Hf = page_wh
        self.Wf, self.Hf = Wf, Hf
        self.W, self.H = out_wh
        self.aspect = Hf / Wf
        st = strength
        # --- curl along X: slope angle theta(s), s in [0,1] (arc length) ----
        K = 6
        knots = np.linspace(0, 1, K)
        amp = rng.uniform(0.1, 0.55) * st
        theta_k = rng.normal(0, amp, K)
        if rng.random() < 0.5:  # book-spine style strong curl near one side
            side = rng.integers(2)
            theta_k[0 if side == 0 else -1] += rng.choice([-1, 1]) * rng.uniform(0.3, 0.8) * st
        self._theta = CubicSpline(knots, theta_k)
        s = np.linspace(0, 1, 2001)
        th = self._theta(s)
        ds = s[1] - s[0]
        self._s = s
        self._x3 = np.concatenate([[0], np.cumsum(np.cos(th[:-1]) * ds)])
        self._z3 = np.concatenate([[0], np.cumsum(np.sin(th[:-1]) * ds)])
        # --- mild bend along Y (adds non-developable component) -------------
        self._ybend = rng.uniform(-0.04, 0.04) * st
        # --- bumps ----------------------------------------------------------
        nb = rng.integers(0, 4)
        self._bumps = [(rng.uniform(0, 1), rng.uniform(0, self.aspect), rng.uniform(0.08, 0.25),
                        rng.uniform(-0.015, 0.015) * st) for _ in range(nb)]
        # --- pose & camera ---------------------------------------------------
        self.R = _rot(np.deg2rad(rng.uniform(-22, 22) * st), np.deg2rad(rng.uniform(-22, 22) * st),
                      np.deg2rad(rng.uniform(-12, 12) * st))
        self.f = rng.uniform(0.75, 1.3) * max(self.W, self.H)
        self.c = np.array([self.W / 2, self.H / 2])
        self.D = 1.5
        self._center = np.array([self._x3[-1] / 2, self.aspect / 2, np.mean(self._z3)])
        fill = rng.uniform(0.72, 0.93)
        for _ in range(4):
            p = self._project_norm(self._outline())
            span = np.maximum(p.max(0) - p.min(0), 1e-6)
            r = max(span[0] / (fill * self.W), span[1] / (fill * self.H))
            self.D *= r
        p = self._project_norm(self._outline())
        lo, hi = p.min(0), p.max(0)
        free = np.array([self.W, self.H]) - (hi - lo)
        off = np.array([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)]) * np.maximum(free, 0)
        self.shift = off - lo

    def _outline(self):
        t = np.linspace(0, 1, 50)
        X = np.concatenate([t, np.ones_like(t), t[::-1], np.zeros_like(t)])
        Y = np.concatenate([np.zeros_like(t), t, np.ones_like(t), t[::-1]]) * self.aspect
        return np.stack([X, Y], 1)

    def surface(self, XY):
        """XY in page units (width = 1) -> 3-D points in page frame."""
        X, Y = XY[:, 0], XY[:, 1]
        x3 = np.interp(X, self._s, self._x3, left=None, right=None)
        z3 = np.interp(X, self._s, self._z3)
        # extrapolate linearly outside [0,1]
        z = z3 + self._ybend * ((Y / self.aspect - 0.5) ** 2) * 4
        for bx, by, r, a in self._bumps:
            z = z + a * np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / (r * r))
        return np.stack([x3, Y, z], 1) - self._center

    def _project_norm(self, XY):
        P = self.surface(XY) @ self.R.T
        P[:, 2] += self.D
        return self.f * P[:, :2] / P[:, 2:3] + self.c

    def forward(self, pts_px):
        """flat page pixels -> photo pixels."""
        pts = np.asarray(pts_px, float).reshape(-1, 2) / self.Wf
        return self._project_norm(pts) + self.shift

    def shading(self, XY, light):
        th = self._theta(np.clip(XY[:, 0], 0, 1))
        n = np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], 1) @ self.R.T
        return np.abs(n @ light)


def _background(rng, H, W):
    base = np.array(rng.choice([[60, 45, 35], [120, 95, 70], [40, 40, 45], [150, 150, 150], [90, 70, 60],
                                [200, 190, 175]]), float)
    base = base * rng.uniform(0.8, 1.2)
    low = cv2.resize(rng.normal(0, 1, (8, 8)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    hi = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 2)
    if rng.random() < 0.5:  # wood-grain like streaks
        streak = cv2.resize(rng.normal(0, 1, (1, 64)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        hi = hi + 2 * streak
    tex = 18 * low + 8 * hi
    return np.clip(base[None, None, :] + tex[..., None], 0, 255)


def simulate_photo(flat: np.ndarray, ann: list[dict], rng: np.random.Generator, out_wh=(1200, 1600),
                   strength=1.0, degrade=True):
    Hf, Wf = flat.shape[:2]
    W, H = out_wh
    pw = PageWarp((Wf, Hf), out_wh, rng, strength)

    # dense flat grid -> photo; invert with Delaunay linear interpolation
    st = 5
    gx, gy = np.meshgrid(np.arange(-st, Wf + 2 * st, st, dtype=float), np.arange(-st, Hf + 2 * st, st, dtype=float))
    flat_pts = np.stack([gx.ravel(), gy.ravel()], 1)
    img_pts = pw.forward(flat_pts)
    inv = LinearNDInterpolator(img_pts, flat_pts)
    q = 2
    qx, qy = np.meshgrid(np.arange(0, W, q, dtype=float), np.arange(0, H, q, dtype=float))
    bmap_lo = inv(qx, qy).astype(np.float32)  # (H/q, W/q, 2), NaN outside
    nan = np.isnan(bmap_lo[..., 0])
    bm = np.where(np.isnan(bmap_lo), -1e4, bmap_lo)
    bmap = cv2.resize(bm, (W, H), interpolation=cv2.INTER_LINEAR)
    mapx, mapy = bmap[..., 0], bmap[..., 1]

    # page appearance in flat space
    paper = np.array([rng.uniform(225, 255), rng.uniform(220, 250), rng.uniform(200, 240)])
    ink = np.array([rng.uniform(0, 50)] * 3) + rng.uniform(0, 15, 3)
    g = flat.astype(np.float32) / 255.0
    page_rgb = paper[None, None] * g[..., None] + ink[None, None] * (1 - g[..., None])
    if degrade:
        light = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1.0])
        light /= np.linalg.norm(light)
        fx, fy = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, pw.aspect, 90))
        sh = pw.shading(np.stack([fx.ravel(), fy.ravel()], 1), light).reshape(fx.shape)
        amb = rng.uniform(0.35, 0.6)
        sh = amb + (1 - amb) * sh / max(sh.max(), 1e-6)
        sh = cv2.resize(sh.astype(np.float32), (Wf, Hf), interpolation=cv2.INTER_CUBIC)
        page_rgb = page_rgb * sh[..., None]

    page = cv2.remap(page_rgb.astype(np.float32), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.remap(np.ones((Hf, Wf), np.float32), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)
    bg = _background(rng, H, W) if degrade else np.full((H, W, 3), 80.0)
    out = page * mask[..., None] + bg * (1 - mask[..., None])

    occl = np.zeros((H, W), np.float32)
    if degrade:
        # global illumination gradient + vignette
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        a, b = rng.uniform(-0.35, 0.35, 2)
        grad = 1 + a * (xx / W - 0.5) + b * (yy / H - 0.5)
        r2 = ((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2
        vig = 1 - rng.uniform(0, 0.5) * r2
        out *= (grad * vig)[..., None]
        # cast shadow (e.g. phone / hand)
        if rng.random() < 0.6:
            sm = np.zeros((H, W), np.float32)
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            pts = np.stack([cx + rng.normal(0, W * 0.35, 6), cy + rng.normal(0, H * 0.35, 6)], 1)
            cv2.fillPoly(sm, [cv2.convexHull(pts.astype(np.int32))], 1.0)
            sm = cv2.GaussianBlur(sm, (0, 0), rng.uniform(15, 60))
            out *= (1 - rng.uniform(0.25, 0.5) * sm)[..., None]
        # finger occlusion from a border
        if rng.random() < 0.4:
            side = rng.integers(4)
            L = rng.uniform(0.12, 0.25) * min(W, H)
            wid = rng.uniform(0.05, 0.08) * min(W, H)
            t = rng.uniform(0.2, 0.8)
            if side == 0:
                c, ang = (t * W, 0), 90
            elif side == 1:
                c, ang = (W, t * H), 180
            elif side == 2:
                c, ang = (t * W, H), 270
            else:
                c, ang = (0, t * H), 0
            ang += rng.uniform(-30, 30)
            fm = np.zeros((H, W), np.float32)
            cv2.ellipse(fm, (int(c[0]), int(c[1])), (int(L), int(wid)), ang, 0, 360, 1.0, -1)
            fm = cv2.GaussianBlur(fm, (0, 0), 2)
            skin = np.array([rng.uniform(170, 230), rng.uniform(120, 170), rng.uniform(90, 140)])
            out = out * (1 - fm[..., None]) + skin[None, None] * fm[..., None] * (0.8 + 0.2 * fm[..., None])
            occl = fm
        sigma = rng.uniform(0.0, 1.1)
        if sigma > 0.3:
            out = cv2.GaussianBlur(out, (0, 0), sigma)
        out = out + rng.normal(0, rng.uniform(1.5, 5), out.shape)
    out = np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8)[..., ::-1])  # RGB -> BGR
    if degrade:
        q = int(rng.integers(55, 92))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    warped_ann = []
    for a in ann:
        b = dict(a)
        b["pts"] = pw.forward(np.asarray(a["pts"], float)).tolist()
        warped_ann.append(b)
    bmap_lo[nan] = np.nan
    return {
        "image": out,                       # BGR uint8 (OpenCV convention)
        "ann": warped_ann,                  # annotations in photo pixels
        "page_mask": (mask > 0.5).astype(np.uint8),
        "occlusion": (occl > 0.5).astype(np.uint8),
        "bmap_lo": bmap_lo,                 # photo(stride 2) -> flat px, NaN off-page
        "warp": pw,
    }
