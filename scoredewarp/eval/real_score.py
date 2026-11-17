"""Diagnostics on frozen input staff traces; these are not manual ground truth."""
import numpy as np


def trace_metrics(staves, masks):
    """Evaluate mapped Nx5x2 staff samples on explicit valid supports.

    Y standard deviation is normalized by mapped adjacent-line spacing. At least
    eight columns and half of a staff's source columns must survive. Coverage is
    reported even when straightness cannot be measured.
    """
    if len(staves) != len(masks):
        raise ValueError('One validity mask is required per staff')
    total = kept = 0
    straight, spacing_cv, inverted = [], [], []
    usable = 0
    for points, valid in zip(staves, masks):
        points, valid = np.asarray(points), np.asarray(valid, bool)
        if points.ndim != 3 or points.shape[1:] != (5, 2) or valid.shape != points.shape[:2]:
            raise ValueError('Expected Nx5x2 points and Nx5 masks')
        valid = valid & np.isfinite(points).all(-1)
        total += valid.size
        kept += int(valid.sum())
        columns = valid.all(1)
        if columns.sum() < max(8, int(np.ceil(len(points)*.5))):
            continue
        p = points[columns]
        spacing = np.diff(p[:, :, 1], axis=1)
        scale = float(np.median(spacing))
        if not np.isfinite(scale) or scale <= 1e-6:
            continue
        straight.extend((np.std(p[:, :, 1], axis=0)/scale).tolist())
        spacing_cv.append(float(np.std(spacing)/scale))
        adjacent = columns[:-1] & columns[1:]
        if adjacent.any():
            inverted.extend((np.diff(points[:, :, 0], axis=0)[adjacent] <= 0).ravel().tolist())
        usable += 1
    return {'point_coverage': kept/total if total else None, 'source_points': total,
            'retained_points': kept, 'usable_staves': usable, 'source_staves': len(staves),
            'staff_straight': float(np.median(straight)) if straight else None,
            'staff_spacing_cv': float(np.median(spacing_cv)) if spacing_cv else None,
            'horizontal_reversal_rate': float(np.mean(inverted)) if inverted else None}


def margin_content_points(ink, staves, *, stride=3, max_points=20000, seed=0):
    """Sample ink pixels above the first and below the last frozen staff.

    These bands hold piece numbers, titles, tempo marks and page numbers that a
    staff-only diagnostic never sees. The band height is the median vertical
    gap between consecutive staves (one gap above and one below), so it scales
    with the page. Ink in the bands is a proxy for content that should survive,
    not a manual annotation; it may include stray marks outside the page.
    """
    if len(staves) < 2:
        return np.zeros((0, 2))
    tops = np.array([np.median(np.asarray(s)[:, 0, 1]) for s in staves])
    bottoms = np.array([np.median(np.asarray(s)[:, 4, 1]) for s in staves])
    order = np.argsort(tops)
    tops, bottoms = tops[order], bottoms[order]
    gap = float(np.median(np.diff(tops)))
    xs = np.concatenate([np.asarray(s)[..., 0].ravel() for s in staves])
    x0, x1 = np.percentile(xs, [1, 99])
    h, w = ink.shape[:2]
    x0, x1 = int(max(0, x0)), int(min(w, x1))
    bands = [(max(0, tops[0]-gap), tops[0]-0.5*(bottoms[0]-tops[0])),
             (bottoms[-1]+0.5*(bottoms[-1]-tops[-1]), min(h, bottoms[-1]+gap))]
    points = []
    for y0, y1 in bands:
        y0, y1 = int(max(0, y0)), int(min(h, y1))
        if y1 <= y0 or x1 <= x0:
            continue
        sub = ink[y0:y1:stride, x0:x1:stride] > 0
        yy, xx = np.nonzero(sub)
        points.append(np.stack([xx*stride+x0, yy*stride+y0], 1))
    points = np.concatenate(points) if points else np.zeros((0, 2))
    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    return points.astype(float)


def sampled_map_folds(bmap):
    """Nonpositive Jacobian rate among finite backward-grid cells, not a proof."""
    field = np.asarray(bmap.grid, float)
    dx, dy = field[:-1, 1:]-field[:-1, :-1], field[1:, :-1]-field[:-1, :-1]
    determinant = dx[..., 0]*dy[..., 1]-dx[..., 1]*dy[..., 0]
    finite = (np.isfinite(field[:-1, :-1]).all(-1) & np.isfinite(field[1:, :-1]).all(-1) &
              np.isfinite(field[:-1, 1:]).all(-1) & np.isfinite(field[1:, 1:]).all(-1))
    return {'finite_grid_cell_fraction': float(finite.mean()),
            'nonpositive_jacobian_fraction': float((determinant[finite] <= 0).mean()) if finite.any() else None}
