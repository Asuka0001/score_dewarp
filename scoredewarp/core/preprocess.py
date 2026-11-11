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
