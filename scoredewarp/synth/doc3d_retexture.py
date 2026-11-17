"""Render flat score pages onto real captured paper deformations (Doc3D UV).

Doc3D stores, for every photo pixel, the texture coordinate (u, v) of the
paper point seen there. The meshes were captured from real paper, so curl
direction, corner folds and local wrinkles are not restricted to any
parametric family -- unlike our own simulator, whose developable curl along X
is the same family the geometric fitter assumes. Texturing a music page with
the UV field therefore gives an evaluation set that is independent of both.

Conventions follow ``scripts/prepare_datasets.doc3d_uv_to_map``: OpenCV reads
the EXR as BGR = (validity, V, U); flat Y = 1 - V.

The Doc3D texture is mapped to the whole unit square regardless of the
physical sheet's aspect ratio, so the global aspect of the ground-truth flat
page is not physically meaningful. Evaluation must therefore be invariant to a
global anisotropic scale (the DocUNet protocol likewise resizes outputs to the
reference size).
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


def read_doc3d_uv(path):
    uv = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if uv is None or uv.ndim != 3 or uv.shape[2] < 3:
        raise ValueError(f"Unreadable Doc3D UV: {path}")
    valid = (uv[..., 0] > 0.5) & np.isfinite(uv[..., :3]).all(-1)
    coords = np.stack([uv[..., 2], 1.0 - uv[..., 1]], -1).astype(np.float64)
    valid &= ((coords >= -1e-4) & (coords <= 1.0001)).all(-1)
    coords = np.clip(coords, 0, 1)
    coords[~valid] = np.nan
    return coords, valid


def uv_statistics(coords, valid, grid=48):
    """Coverage of the flat page and non-planarity of the deformation.

    ``flat_coverage``: fraction of the unit square that is visible.
    ``nonplanarity``: RMS residual (in % of the visible page diagonal) of the
    best homography from flat to photo -- 0 for a flat sheet under any
    perspective, large for strong curl, folds and wrinkles.
    """
    h, w = valid.shape
    yy, xx = np.nonzero(valid)
    uv = coords[yy, xx]
    hist = np.zeros((grid, grid), bool)
    gi = np.minimum((uv * grid).astype(int), grid - 1)
    hist[gi[:, 1], gi[:, 0]] = True
    stats = {"page_fraction": float(valid.mean()), "flat_coverage": float(hist.mean())}
    if len(uv) < 100:
        stats["nonplanarity"] = float("inf")
        return stats
    sel = np.random.default_rng(0).choice(len(uv), min(4000, len(uv)), replace=False)
    src = uv[sel].astype(np.float32)
    dst = np.stack([xx[sel], yy[sel]], 1).astype(np.float32)
    H, _ = cv2.findHomography(src, dst, 0)
    pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    diag = np.hypot(np.ptp(dst[:, 0]), np.ptp(dst[:, 1]))
    stats["nonplanarity"] = float(np.sqrt(((pred - dst) ** 2).sum(1).mean()) / max(diag, 1) * 100)
    return stats


def _crop_window(valid, rng, out_size):
    """Frame the sheet like a phone photo: page bbox + random margin.

    Returns (x0, y0, scale, (W, H)): output pixel q maps to source pixel
    (x0 + q.x / scale, y0 + q.y / scale) on the Doc3D grid.
    """
    ys, xs = np.nonzero(valid)
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    m = rng.uniform(0.04, 0.14, 4) * max(bx1 - bx0, by1 - by0)
    x0, x1, y0, y1 = bx0 - m[0], bx1 + m[1], by0 - m[2], by1 + m[3]
    scale = out_size / max(x1 - x0, y1 - y0)
    even = lambda v: 2 * int(round(v * scale / 2))  # stride-2 GT grid stays aligned
    return x0, y0, scale, (even(x1 - x0), even(y1 - y0))


def _resample(field, x0, y0, scale, wh, interp=cv2.INTER_LINEAR, border=0):
    W, H = wh
    qx, qy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # pixel-centre convention: output centre q+.5 <-> source centre p+.5
    mx = (x0 + (qx + 0.5) / scale - 0.5).astype(np.float32)
    my = (y0 + (qy + 0.5) / scale - 0.5).astype(np.float32)
    return cv2.remap(field, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _upsample_field(coords, valid, window):
    """Mask-normalised bilinear resampling, so NaNs do not bleed into the page."""
    filled = np.where(valid[..., None], coords, 0.0).astype(np.float32)
    up = _resample(filled, *window)
    wu = _resample(valid.astype(np.float32), *window)
    mask = wu > 0.5
    up = up / np.maximum(wu, 1e-6)[..., None]
    up[~mask] = np.nan
    return up, mask, wu


def _shading(image, valid):
    """Illumination of the Doc3D page with its printed texture removed."""
    lum = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    page = np.where(valid, lum, 0).astype(np.float32)
    closed = cv2.morphologyEx(page, cv2.MORPH_CLOSE, k, iterations=2)
    closed = cv2.dilate(closed, np.ones((3, 3), np.uint8))  # pull bright paper over text
    closed = cv2.GaussianBlur(closed, (0, 0), 2.0)
    ref = np.percentile(closed[valid], 97) if valid.any() else 1.0
    return np.clip(closed / max(ref, 1.0), 0.15, 1.1)


def render(flat_gray, flat_ann, uv_path, img_path, rng, out_size=1600):
    """Return photo (BGR), stride-2 photo->flat map (flat px, NaN off page),
    annotations in photo px (NaN where the flat point is not visible) and stats."""
    coords, valid = read_doc3d_uv(uv_path)
    src = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if src is None or src.shape[:2] != valid.shape:
        raise ValueError(f"Doc3D image/UV mismatch: {img_path}")
    Hf, Wf = flat_gray.shape[:2]
    scale = np.array([Wf, Hf], float)
    window = _crop_window(valid, rng, out_size)
    x0, y0, s, (W, H) = window
    up, mask, wu = _upsample_field(coords, valid, window)

    # --- appearance -------------------------------------------------------
    paper = (rng.uniform(215, 252) + np.array([rng.uniform(-4, 6), rng.uniform(-4, 4), rng.uniform(-20, 2)])).astype(np.float32)
    ink = (rng.uniform(5, 50) + rng.uniform(0, 12, 3)).astype(np.float32)
    g = flat_gray.astype(np.float32) / 255.0
    page_rgb = paper[None, None] * g[..., None] + ink[None, None] * (1 - g[..., None])
    mx = np.where(mask, up[..., 0] * Wf, -10).astype(np.float32)
    my = np.where(mask, up[..., 1] * Hf, -10).astype(np.float32)
    page = cv2.remap(page_rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    shade = _resample(_shading(src, valid).astype(np.float32), *window, interp=cv2.INTER_CUBIC, border=1.0)
    page *= shade[..., None]
    bg = _resample(src[..., ::-1].astype(np.float32), *window)
    bg = np.where(bg.sum(-1, keepdims=True) > 0, bg, 40.0)  # outside the Doc3D frame
    alpha = np.clip((wu - 0.25) / 0.5, 0, 1)[..., None]
    out = page * alpha + bg * (1 - alpha)
    sigma = rng.uniform(0.0, 0.9)
    if sigma > 0.3:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    out = out + rng.normal(0, rng.uniform(1.0, 4.0), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)[..., ::-1]
    ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(out), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 93))])
    photo = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # --- ground truth -----------------------------------------------------
    bmap = up[::2, ::2] * scale  # photo (stride 2) -> flat px
    yy, xx = np.nonzero(valid)
    flat_pts = coords[yy, xx] * scale
    photo_pts = (np.stack([xx, yy], 1) - np.array([x0, y0]) + 0.5) * s - 0.5
    n = valid.shape[0]
    inv = LinearNDInterpolator(flat_pts, photo_pts)
    tree = cKDTree(flat_pts)
    max_gap = 2.0 * max(Wf, Hf) / n
    photo_ann = []
    for a in flat_ann:
        p = np.asarray(a["pts"], float).reshape(-1, 2)
        q = inv(p)
        d, _ = tree.query(p)
        q[d > max_gap] = np.nan
        b = dict(a)
        b["pts"] = [[None if not np.isfinite(v) else float(v) for v in row] for row in q]
        photo_ann.append(b)
    stats = uv_statistics(coords, valid)
    return photo, bmap.astype(np.float32), photo_ann, stats
