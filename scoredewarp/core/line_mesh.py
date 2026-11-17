"""Line-guided, bounded document mesh refinement (inspired by RDGR).

Fit entire staff lines to absolute horizontal, equally spaced targets rather
than accumulating short pairwise differences. A coarse displacement grid has
thin-plate bending regularization and Huber outlier weighting. Only the vertical
residual is observable from horizontal lines: leave X unchanged. This avoids
inventing horizontal stretch from line evidence alone.

Top/bottom are fixed, and every bilinear cell is checked for bounded vertical
scale and shear. The accepted map is column-monotone and is inverted exactly
per column; no Delaunay inversion, convex-hull holes or content-derived crop.
This is a 2D residual method, not a reproduction of RDGR or UVDoc.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

from .field import _Grid
from .transforms import BackwardMap


class _RectGrid(_Grid):
    def __init__(self, width, height, node):
        nx = max(3, int(np.ceil((width - 1) / node)) + 1)
        ny = max(3, int(np.ceil((height - 1) / node)) + 1)
        super().__init__(nx - 1, ny - 1, 1)
        self.step = np.array([(width - 1) / (nx - 1), (height - 1) / (ny - 1)])

    def weights(self, pts):
        return super().weights(np.asarray(pts) / self.step)


def _observations(structures):
    height, width = structures.shape
    points, targets, scales = [], [], []
    rejected = 0
    for staff in structures.staves:
        p = np.asarray(staff.tuples, dtype=float)
        if p.ndim != 3 or p.shape[1:] != (5, 2) or len(p) < 8 or not np.isfinite(p).all():
            rejected += 1
            continue
        spacing = np.diff(p[..., 1], axis=1)
        scale = float(np.median(spacing))
        if (scale < 2 or np.any(spacing <= 0) or np.std(spacing) / scale > .25
                or np.ptp(p[:, 0, 0]) < .15 * width
                or np.any(p < 0) or np.any(p[..., 0] > width - 1) or np.any(p[..., 1] > height - 1)):
            rejected += 1
            continue
        # Keep this system's position/size; do not force different instruments
        # or small cue staves to share a global spacing or inter-system gap.
        levels = np.arange(5) * scale
        top = np.median(p[..., 1] - levels)
        target = np.broadcast_to(top + levels, p.shape[:2])
        points.append(p.reshape(-1, 2))
        targets.append(target.ravel())
        scales.append(scale)
    return points, targets, scales, rejected


def mesh_bounds(displacement, step):
    """All bilinear-cell corner derivatives: bounds hold inside each cell too."""
    dx = np.diff(displacement, axis=1) / step[0]
    dy = 1 + np.diff(displacement, axis=0) / step[1]
    return {'vertical_scale_min': float(dy.min()), 'vertical_scale_max': float(dy.max()),
            'vertical_shear_max': float(np.abs(dx).max())}


def fit_line_mesh(structures, node=32, bending=2., identity=.015,
                  min_scale=.5, max_scale=2., max_shear=.5):
    """Return (backward residual map or None, diagnostics).

    Bounds apply to this residual, not to an arbitrary upstream learned map.
    A full-photo bijection is obtained by U=x, V_y>0 and V(top/bottom)=top/bottom.
    Refuse a solution that improves the median by damaging one of its systems.
    """
    if node <= 0 or bending < 0 or identity <= 0 or not 0 < min_scale < 1 < max_scale or max_shear <= 0:
        raise ValueError('Invalid line mesh configuration')
    h, w = structures.shape
    points, targets, scales, rejected = _observations(structures)
    info = {'accepted': False, 'staff_systems': len(points), 'rejected_systems': rejected,
            'representation': 'bounded bilinear vertical residual; fixed top/bottom; X unchanged'}
    if not points:
        info['reason'] = 'no reliable staff systems'
        return None, info
    g = _RectGrid(w, h, node)
    pts = np.concatenate(points)
    target = np.concatenate(targets)
    delta = target - pts[:, 1]
    A = g.weights(pts)
    # Eliminate boundary variables rather than weakly penalizing their motion.
    free = np.arange(g.n).reshape(g.ny, g.nx)[1:-1].ravel()
    A = A[:, free]
    smooth = g.smooth_ops()[:, free]
    regularizer = bending ** 2 * (smooth.T @ smooth) + identity ** 2 * eye(len(free))
    robust = np.ones(len(pts))
    huber = max(.75, .15 * np.median(scales))
    for _ in range(5):
        weighted = diags(robust)
        solution = spsolve((A.T @ weighted @ A + regularizer).tocsc(), A.T @ (robust * delta))
        residual = np.abs(A @ solution - delta)
        robust = np.minimum(1., huber / np.maximum(residual, 1e-9))
    d = np.zeros(g.n)
    d[free] = solution
    d = d.reshape(g.ny, g.nx)
    if not np.isfinite(d).all():
        info['reason'] = 'nonfinite solve'
        return None, info
    alpha = 1.
    while alpha >= 1 / 64:
        bounds = mesh_bounds(alpha * d, g.step)
        if (bounds['vertical_scale_min'] >= min_scale and bounds['vertical_scale_max'] <= max_scale
                and bounds['vertical_shear_max'] <= max_shear):
            break
        alpha *= .5
    else:
        info['reason'] = 'scale/shear bounds'
        return None, info
    d *= alpha
    correction = g.weights(pts) @ d.ravel()
    offsets = np.cumsum([0] + [len(p) for p in points])
    before, after = [], []
    for i, scale in enumerate(scales):
        sl = slice(offsets[i], offsets[i + 1])
        old = pts[sl, 1].reshape(-1, 5)
        new = (pts[sl, 1] + correction[sl]).reshape(-1, 5)
        before.append(float(np.median(np.std(old, axis=0)) / scale))
        after.append(float(np.median(np.std(new, axis=0)) / scale))
    info.update(bounds, damping=alpha, straight_before=before, straight_after=after,
                control_shape=[g.ny, g.nx], max_displacement_px=float(np.abs(d).max()))
    if np.median(after) >= np.median(before) - .002 or np.any(np.array(after) > np.array(before) + .05):
        info['reason'] = 'no safe structural improvement'
        return None, info

    xs = np.arange(g.nx) * g.step[0]
    ys = np.arange(g.ny) * g.step[1]

    def inverse(query):
        # The forward mesh is linear in Y between nodes at each fixed X.
        # Invert these monotone columns, preserving the rectangular domain.
        out = query.copy()
        for x in np.unique(query[:, 0]):
            selected = query[:, 0] == x
            xc = np.clip(x, 0, w - 1)
            ix = min(int(xc / g.step[0]), g.nx - 2)
            fraction = (xc - xs[ix]) / g.step[0]
            forward_y = ys + (1 - fraction) * d[:, ix] + fraction * d[:, ix + 1]
            out[selected, 1] = np.interp(query[selected, 1], forward_y, ys)
            # Extend with the boundary cell's slope. This keeps the exact
            # fixed boundary even when H-1 lies between two sampled nodes.
            above = selected & (query[:, 1] < 0)
            below = selected & (query[:, 1] > h - 1)
            out[above, 1] = query[above, 1] * (ys[1] - ys[0]) / (forward_y[1] - forward_y[0])
            out[below, 1] = h - 1 + (query[below, 1] - (h - 1)) * (ys[-1] - ys[-2]) / (forward_y[-1] - forward_y[-2])
        return out

    bm = BackwardMap.from_function(inverse, (w, h), stride=4)
    # Validate the sampled inverse as well. The analytic forward derivative
    # bounds above cover every point in the original bilinear cells.
    from ..eval.real_score import sampled_map_folds
    sampled = sampled_map_folds(bm)
    info['sampled_inverse'] = sampled
    if sampled['finite_grid_cell_fraction'] < 1 or sampled['nonpositive_jacobian_fraction'] > 0:
        info['reason'] = 'invalid sampled inverse'
        return None, info
    info['accepted'] = True
    info['reason'] = 'improved with bounded scale and shear'
    return bm, info
