"""Geometry metrics independent of music annotations.

Evaluate on a fixed regular grid in the input photo. AD removes only global
translation, rotation and uniform scale; shear and nonuniform scale remain.
All errors are normalized by flat-page width. Missing predictions are reported
as coverage, never silently assigned zero error.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .metrics import _similarity_fit


def sample_geometry(folder, max_points=4096):
    folder = Path(folder)
    meta = json.loads((folder / "gt.json").read_text())["meta"]
    image = cv2.imread(str(folder / "photo.jpg"))
    if image is None:
        raise ValueError(f"Unreadable image: {folder}")
    mapping = np.load(folder / "gt_bmap_s2.npy").astype(np.float64)
    h, w = mapping.shape[:2]
    stride = max(1, int(np.ceil(np.sqrt(h * w / max_points))))
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    flat = mapping[::stride, ::stride] / meta["size"][0]
    aspect = meta["size"][1] / meta["size"][0]
    valid = (np.isfinite(flat).all(-1) & (flat >= 0).all(-1) &
             (flat[..., 0] <= 1) & (flat[..., 1] <= aspect))
    photo_points = np.stack([xx * image.shape[1] / w, yy * image.shape[0] / h], -1)
    return image, meta, photo_points, flat, valid


def geometry_metrics(target, predicted, valid, predicted_valid=None):
    target, predicted = np.asarray(target), np.asarray(predicted)
    valid = np.asarray(valid, bool)
    ok = valid & np.isfinite(predicted).all(-1)
    if predicted_valid is not None:
        ok &= predicted_valid
    n = int(valid.sum())
    result = {"coverage": float(ok.sum() / n) if n else 0., "gt_points": n,
              "ad_pct": None, "local_distortion_pct": None}
    if ok.sum() < 6:
        return result
    p, t = predicted[ok], target[ok]
    if ((p - p.mean(0)) ** 2).sum() < 1e-12:
        return result  # a collapsed field cannot be rectified
    scale, rotation, offset = _similarity_fit(p, t)
    if not np.isfinite(scale) or scale <= 1e-12:
        return result
    aligned = scale * predicted @ rotation.T + offset
    result["ad_pct"] = float(np.sqrt(((aligned[ok] - t) ** 2).sum(-1).mean()) * 100)
    errors, lengths = [], []
    for axis in (0, 1):
        mask = np.take(ok, range(ok.shape[axis] - 1), axis=axis) & np.take(ok, range(1, ok.shape[axis]), axis=axis)
        dt, dp = np.diff(target, axis=axis), np.diff(aligned, axis=axis)
        errors.extend(((dp - dt)[mask] ** 2).sum(-1))
        lengths.extend((dt[mask] ** 2).sum(-1))
    if lengths and np.mean(lengths) > 1e-12:
        result["local_distortion_pct"] = float(np.sqrt(np.mean(errors) / np.mean(lengths)) * 100)
    return result


def network_points(out, photo_points, photo_size, aspect):
    field = out["fwd"][0].detach().cpu().numpy().transpose(1, 2, 0)
    prob = out["page"][0, 0].detach().sigmoid().cpu().numpy()
    h, w = field.shape[:2]
    xy = (photo_points / np.array(photo_size) * [w, h]).astype(np.float32)
    coords = cv2.remap(field, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE) * [1, aspect]
    page = cv2.remap(prob, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE) > .5
    return coords, page


def orientation_error_rate(target, predicted, valid):
    """Fraction of sampled page cells whose Jacobian orientation differs from GT.

    Uses all GT page cells, independent of predicted page confidence. This is a
    sampled indicator of folds/collapses, not a guarantee of global invertibility.
    """
    ok = valid & np.isfinite(predicted).all(-1)
    cells = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    def determinant(field):
        dx = field[:-1, 1:] - field[:-1, :-1]
        dy = field[1:, :-1] - field[:-1, :-1]
        return dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    gt_det, pred_det = determinant(target), determinant(predicted)
    cells &= np.abs(gt_det) > 1e-12
    return float((gt_det[cells] * pred_det[cells] <= 0).mean()) if cells.any() else None
