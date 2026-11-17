"""Illumination normalisation and binarisation."""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_illumination(gray: np.ndarray, ksize: int | None = None) -> np.ndarray:
    """Divide by an estimate of the paper background (max-filter + median).

    Removes shading from curl, cast shadows and vignetting; large dark blobs
    (fingers, background) become bright and therefore drop out of the ink mask.
    """
    H, W = gray.shape
    k = ksize or max(15, int(min(H, W) / 60))
    k |= 1
    small = 1.0
    g = gray
    if min(H, W) > 1200:  # background is smooth: estimate at lower resolution
        small = 1200 / min(H, W)
        g = cv2.resize(gray, None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
        k = max(15, int(k * small)) | 1
    bg = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, min(k, 255) if k <= 255 else 255)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    if small != 1.0:
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
    norm = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 255
    return np.clip(norm, 0, 255).astype(np.uint8)


def binarize(norm: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Return ink mask (uint8 0/1)."""
    vals = norm[valid > 0] if valid is not None else norm.ravel()
    if vals.size == 0:
        vals = norm.ravel()
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = min(float(t), 200.0)
    ink = (norm < t).astype(np.uint8)
    if valid is not None:
        ink &= (valid > 0).astype(np.uint8)
    # remove salt noise
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) | (ink & cv2.erode(ink, np.ones((1, 3), np.uint8)))
    return ink


def clean_page(img: np.ndarray, paper: np.ndarray | None = None, spacing: float | None = None,
               mode: str = "gray") -> np.ndarray:
    """Reading copy of a rectified page: shading removed, paper white, ink dark.

    Background = grey-level closing with a kernel just wider than the thickest
    strokes (noteheads, beams ~ one staff spacing), then a median of the same
    size. Unlike a large max filter + blur, a closing keeps the position of
    sharp cast-shadow borders, so no halo of the shadow outline is left.
    ``paper`` (uint8 mask, same size, see ``paper_region``) whitens everything
    off the sheet.
    ``mode``: "gray" (contrast-stretched grey) or "binary" (Otsu on the result).
    """
    gray = to_gray(img)
    H, W = gray.shape
    sp = float(spacing) if spacing and spacing > 0 else max(10.0, min(H, W) / 120)
    sc = min(1.0, 600 / min(H, W) if min(H, W) > 1200 else 1.0)
    g = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else gray
    k = max(7, int(round(1.6 * sp * sc))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, ker)
    bg = cv2.medianBlur(bg, min(k, 255))
    if sc < 1:
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
    norm = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 255
    norm = np.clip(norm, 0, 255)
    on = paper > 0 if paper is not None else np.ones_like(gray, bool)
    vals = norm[on]
    if vals.size > 100:
        # stretch: faint ink -> dark, paper texture -> white
        dark = np.percentile(vals[vals < 200], 5) if (vals < 200).sum() > 50 else 0.0
        lo, hi = min(dark, 120.0), 235.0
        norm = np.clip((norm - lo) / max(hi - lo, 1) * 255, 0, 255)
    out = norm.astype(np.uint8)
    if mode == "binary":
        t, _ = cv2.threshold(out[on].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # halfway from Otsu to white: thin, faint staff lines stay connected
        t = min(t + 0.5 * (235 - t), 215)
        out = np.where(out < t, 0, 255).astype(np.uint8)
    out[~on] = 255
    return out


def paper_region(mask: np.ndarray | None, image: np.ndarray | None = None, band=0.08,
                 dark=0.5) -> np.ndarray | None:
    """Sheet region on the output for ``clean_page``, from a warped paper mask.

    The convex hull fills dents (missed shadowed paper), shrunk a little so the
    dark paper edge is whitened. Background notches of a wavy border lie inside
    the hull; within a band along the border (page margin, no content) the mask
    itself, shrunk, decides, and with ``image`` large dark areas (a table the
    mask leaked into; thin ink is blurred away) are dropped as well.
    """
    if mask is None or mask.mean() / 255 < 0.15:
        return None
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(hull, [cv2.convexHull(np.concatenate(cnts))], 255)
    k = max(5, int(0.01 * min(m.shape))) | 1
    kb = max(9, int(band * min(m.shape))) | 1
    ell = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))  # noqa: E731
    hull = cv2.erode(hull, ell(k))
    core = cv2.erode(hull, ell(kb))
    near = cv2.erode(m, ell(k)) > 0
    if image is not None and core.any():
        g = cv2.GaussianBlur(to_gray(image).astype(np.float32), (0, 0), max(2.0, 0.01 * min(m.shape)))
        level = np.percentile(g[core > 0], 75)
        near &= g >= dark * level
    return np.where((core > 0) | ((hull > 0) & near), 255, 0).astype(np.uint8)
