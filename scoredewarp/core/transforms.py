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
