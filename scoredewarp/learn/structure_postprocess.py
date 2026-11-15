"""Standalone structure heatmap filtering in model-input pixel coordinates.

These masks are diagnostics, not calibrated geometric observations for B/C.
Horizontal channels are thinned along Y, vertical channels along X. No image,
ground-truth region, page type or geometry prediction is needed at inference.
"""
import cv2
import numpy as np


def ridge_scores(probabilities, radius=2):
    """Keep local maxima normal to each channel's expected line direction.

    Equal plateaus deterministically keep their bottom/right edge. Zero padding
    allows maxima on the image boundary; channels are never compared together.
    This axis approximation can break steeply tilted/curved lines.
    """
    p = np.asarray(probabilities, dtype=np.float32)
    if p.ndim != 3 or p.shape[0] != 3 or min(p.shape[1:]) < 1:
        raise ValueError("Expected three nonempty C,H,W probability maps")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ValueError("Radius must be a nonnegative integer")
    result = p.copy()
    if radius == 0:
        return result
    for channel in range(3):
        image = p[channel]
        axis = 0 if channel < 2 else 1
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = np.pad(image, padding, constant_values=-np.inf)
        keep = image > 0
        for offset in range(1, radius + 1):
            before = [slice(None), slice(None)]
            after = [slice(None), slice(None)]
            before[axis] = slice(radius-offset, radius-offset+image.shape[axis])
            after[axis] = slice(radius+offset, radius+offset+image.shape[axis])
            keep &= (image >= padded[tuple(before)]) & (image > padded[tuple(after)])
        result[channel] *= keep
    return result


def structure_masks(probabilities, thresholds=(.5, .5, .5), radius=0):
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.shape != (3,) or not np.isfinite(thresholds).all() or ((thresholds <= 0) | (thresholds > 1)).any():
        raise ValueError("Provide three finite thresholds in (0,1]")
    return ridge_scores(probabilities, radius) >= thresholds[:, None, None]


def threshold_counts(scores, truth, thresholds, tolerance=1):
    """Exact and tolerant micro counts for a whole threshold grid in one pass.

    Tolerant precision counts predicted pixels near truth; tolerant recall counts
    truth pixels near predictions. They use different numerators deliberately.
    Histograms avoid building a full image mask for every candidate threshold.
    """
    scores = ridge_scores(scores, 0)
    truth = np.asarray(truth, dtype=bool)
    grid = np.asarray(thresholds, dtype=np.float32)
    if truth.shape != scores.shape:
        raise ValueError("Truth and scores must have the same shape")
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or (grid <= 0).any() or (grid > 1).any() or (np.diff(grid) <= 0).any():
        raise ValueError("Threshold grid must be strictly increasing in (0,1]")
    if not isinstance(tolerance, int) or isinstance(tolerance, bool) or tolerance < 0:
        raise ValueError("Tolerance must be a nonnegative integer")
    kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    bins = np.r_[-np.inf, grid, np.inf]

    def counts(values):
        hist = np.histogram(values, bins=bins)[0][1:]
        return hist[::-1].cumsum()[::-1]

    result = np.zeros((len(grid), 3, 5), dtype=np.int64)
    for channel, (p, t) in enumerate(zip(scores, truth)):
        near_truth = cv2.dilate(t.astype(np.uint8), kernel) > 0
        near_scores = cv2.dilate(p, kernel)
        result[:, channel, 0] = counts(p[t])
        result[:, channel, 1] = counts(p.ravel())
        result[:, channel, 2] = t.sum()
        result[:, channel, 3] = counts(p[near_truth])
        result[:, channel, 4] = counts(near_scores[t])
    return result
