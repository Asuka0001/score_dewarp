"""Backward-map representation shared by all stages.

A BackwardMap stores, on a regular output grid with `stride`, the source
coordinates each output pixel samples from. Stages compose by sampling one map
with another; forward point mapping (source -> output) is done by scattered
inversion and is used for exporting keypoints and for evaluation.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator


class BackwardMap:
    def __init__(self, grid: np.ndarray, stride: int, out_size: tuple):
        self.grid = grid.astype(np.float32)   # (gh, gw, 2) source xy, NaN allowed
        self.stride = stride
        self.out_size = tuple(int(v) for v in out_size)  # (W, H)
        self._inv = None

    # -- constructors -------------------------------------------------------
    @staticmethod
    def from_homography(Hm: np.ndarray, out_size, stride=4):
        W, H = out_size
        gx, gy = np.meshgrid(np.arange(0, W + stride, stride, dtype=np.float64),
                             np.arange(0, H + stride, stride, dtype=np.float64))
        pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 0)
        src = np.linalg.inv(Hm) @ pts
        src = (src[:2] / src[2:]).T.reshape(gx.shape + (2,))
        return BackwardMap(src, stride, out_size)

    @staticmethod
    def from_function(fn, out_size, stride=4):
        W, H = out_size
        gx, gy = np.meshgrid(np.arange(0, W + stride, stride, dtype=np.float64),
                             np.arange(0, H + stride, stride, dtype=np.float64))
        src = fn(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(gx.shape + (2,))
        return BackwardMap(src, stride, out_size)

    def rescale(self, s: float) -> "BackwardMap":
        """Same geometry on an output canvas scaled by ``s`` (still output -> photo)."""
        W, H = self.out_size
        size = (int(round(W * s)), int(round(H * s)))
        return BackwardMap.from_function(lambda q: self.sample(q / s), size, self.stride)

    def local_scale(self):
        """Per grid cell: photo pixels per output pixel (sqrt |det J|), NaN off the photo map."""
        g = np.asarray(self.grid, float)
        dx = (g[:-1, 1:] - g[:-1, :-1]) / self.stride
        dy = (g[1:, :-1] - g[:-1, :-1]) / self.stride
        det = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
        return np.sqrt(np.abs(det))

    # -- use ------------------------------------------------------------------
    def full_maps(self):
        W, H = self.out_size
        fx, fy = np.meshgrid(np.arange(W, dtype=np.float32) / self.stride, np.arange(H, dtype=np.float32) / self.stride)
        g = np.nan_to_num(self.grid, nan=-1e5)
        mx = cv2.remap(np.ascontiguousarray(g[..., 0]), fx, fy, cv2.INTER_LINEAR)
        my = cv2.remap(np.ascontiguousarray(g[..., 1]), fx, fy, cv2.INTER_LINEAR)
        return mx, my

    def apply(self, img, interp=cv2.INTER_CUBIC, border=cv2.BORDER_CONSTANT, value=255):
        mx, my = self.full_maps()
        if img.ndim == 3 and np.isscalar(value):
            value = (value,) * img.shape[2]
        return cv2.remap(img, mx, my, interp, borderMode=border, borderValue=value)

    def sample(self, pts):
        """Source coords of arbitrary output points (bilinear on the grid)."""
        from scipy.ndimage import map_coordinates
        pts = np.asarray(pts, np.float64).reshape(-1, 2) / self.stride
        coords = [pts[:, 1], pts[:, 0]]
        x = map_coordinates(self.grid[..., 0].astype(np.float64), coords, order=1, cval=np.nan)
        y = map_coordinates(self.grid[..., 1].astype(np.float64), coords, order=1, cval=np.nan)
        return np.stack([x, y], 1)

    def compose(self, after: "BackwardMap") -> "BackwardMap":
        """`after` maps out2 -> out1 (= this map's output space); result maps out2 -> src."""
        g = after.grid.reshape(-1, 2)
        src = self.sample(np.nan_to_num(g, nan=-1e5))
        src[np.isnan(g).any(1)] = np.nan
        return BackwardMap(src.reshape(after.grid.shape), after.stride, after.out_size)

    def pad(self, left=0, top=0, right=0, bottom=0) -> "BackwardMap":
        """Grow the output canvas; new pixels map nowhere (rendered as fill)."""
        W, H = self.out_size
        size = (W + int(left) + int(right), H + int(top) + int(bottom))
        shift = BackwardMap.from_function(lambda q: q - [int(left), int(top)], size, self.stride)
        return self.compose(shift)

    def forward_points(self, pts):
        """source -> output coordinates (NaN when outside the mapped area)."""
        if self._inv is None:
            gh, gw = self.grid.shape[:2]
            ox, oy = np.meshgrid(np.arange(gw) * self.stride, np.arange(gh) * self.stride)
            src = self.grid.reshape(-1, 2)
            out = np.stack([ox.ravel(), oy.ravel()], 1).astype(np.float64)
            ok = np.isfinite(src).all(1)
            self._inv = LinearNDInterpolator(src[ok], out[ok])
        pts = np.asarray(pts, float).reshape(-1, 2)
        return self._inv(pts)


def native_scale(bmap: "BackwardMap", photo_size, content=None) -> float:
    """Output scale at which one output pixel covers ~one photo pixel.

    Median of the local scale over the printed content (convex hull of
    ``content`` points, output coords) -- or over all cells that map into the
    photo. The page is sampled unevenly (perspective, curl), so the median
    keeps the typical photo resolution: parts nearer the camera are slightly
    downsampled, farther parts slightly upsampled."""
    ls = bmap.local_scale()
    g = bmap.grid[:-1, :-1]
    ok = np.isfinite(ls) & np.isfinite(g).all(-1)
    ok &= (g >= 0).all(-1) & (g <= np.asarray(photo_size) - 1).all(-1)
    if content is not None and len(content) >= 3:
        hull = cv2.convexHull(np.asarray(content, np.float32))
        gh, gw = ls.shape
        m = np.zeros((gh, gw), np.uint8)
        cv2.fillPoly(m, [np.round((hull.reshape(-1, 2) / bmap.stride)).astype(np.int32)], 1)
        if (ok & (m > 0)).sum() >= 20:
            ok &= m > 0
    return float(np.median(ls[ok])) if ok.any() else 1.0


def photo_footprint(src_pts, photo_size, gap, scale=0.25):
    """Photo area covered by a mapped region, as a test ``inside(pts) -> bool``.

    ``src_pts``: photo coords of the region's grid nodes; ``gap``: their
    spacing in photo pixels (holes up to it are closed). Used to keep an
    extrapolated surface from showing photo pixels a second time where it
    runs back over the page (past a fold)."""
    W0, H0 = photo_size
    foot = np.zeros((int(H0 * scale) + 1, int(W0 * scale) + 1), np.uint8)
    q = np.round(np.asarray(src_pts, float).reshape(-1, 2) * scale)
    q = q[np.isfinite(q).all(1)].astype(int)
    q = q[(q[:, 0] >= 0) & (q[:, 0] < foot.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < foot.shape[0])]
    foot[q[:, 1], q[:, 0]] = 1
    k = 2 * (int(np.ceil(gap * scale)) + 1) + 1
    foot = cv2.morphologyEx(foot, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    def inside(pts):
        p = np.round(np.nan_to_num(np.asarray(pts, float), nan=-1) * scale).astype(int)
        ok = (p[..., 0] >= 0) & (p[..., 0] < foot.shape[1]) & (p[..., 1] >= 0) & (p[..., 1] < foot.shape[0])
        out = np.zeros(p.shape[:-1], bool)
        out[ok] = foot[p[ok][:, 1], p[ok][:, 0]] > 0
        return out
    return inside


def mapping_quality(bmap, photo_size):
    """Check all four bilinear Jacobian corners in finite, in-photo cells.

    Includes the second triangle missed by a top-left-only determinant test.
    This is a local check, not proof of global injectivity or correct geometry.
    Singular values report stretching; they are not forced to unity because
    perspective and output resolution legitimately change the local scale.
    """
    p = np.asarray(bmap.grid, float)
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
    valid = np.ones(a.shape[:2], bool)
    for corner in (a, b, c, d):
        valid &= np.isfinite(corner).all(-1)
        valid &= (corner >= 0).all(-1) & (corner <= np.asarray(photo_size) - 1).all(-1)
    derivatives = [(b-a, c-a), (b-a, d-b), (d-c, c-a), (d-c, d-b)]
    count = int(valid.sum())
    if not count:
        return {'checked_cells': 0, 'folded_cell_fraction': None, 'safe_local_orientation': False}
    jac = np.stack([np.stack([dx[valid], dy[valid]], -1) / bmap.stride for dx, dy in derivatives])
    determinant = np.linalg.det(jac)
    singular = np.linalg.svd(jac, compute_uv=False)
    folds = (determinant <= 0).any(0)
    return {'checked_cells': count, 'folded_cell_fraction': float(folds.mean()),
            'safe_local_orientation': not bool(folds.any()),
            'jacobian_min': float(determinant.min()),
            'local_scale_p01': float(np.quantile(singular[..., 1], .01)),
            'local_scale_p99': float(np.quantile(singular[..., 0], .99)),
            'anisotropy_p99': float(np.quantile(singular[..., 0] / np.maximum(singular[..., 1], 1e-8), .99))}
