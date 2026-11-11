"""Stage A: page boundary detection + homography + deskew (coarse rectification)."""
from __future__ import annotations

import cv2
import numpy as np


def order_quad(q):
    q = np.asarray(q, np.float32).reshape(4, 2)
    s = q.sum(1)
    d = q[:, 1] - q[:, 0]
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]], np.float32)


def detect_page_quad(img: np.ndarray):
    """Largest bright, low-saturation quadrilateral. Returns 4x2 (tl,tr,br,bl) or None."""
    H, W = img.shape[:2]
    sc = 800 / max(H, W)
    small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    S = hsv[..., 1].astype(np.float32)
    score = cv2.GaussianBlur(L - 0.6 * S, (0, 0), 3)
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(5, int(0.02 * max(small.shape[:2])))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c) / (small.shape[0] * small.shape[1])
    if area < 0.15 or area > 0.97:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.12, 23):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
    # the quad must explain the segmented page region (IoU), otherwise segmentation failed
    comp = np.zeros_like(m)
    cv2.drawContours(comp, [c], -1, 255, -1)
    qm = np.zeros_like(m)
    cv2.fillPoly(qm, [order_quad(quad).astype(np.int32)], 255)
    inter = np.logical_and(comp > 0, qm > 0).sum()
    union = np.logical_or(comp > 0, qm > 0).sum()
    if inter / max(union, 1) < 0.85:
        return None
    quad = order_quad(quad) / sc
    return quad if quad_is_plausible(quad, W, H) else None


def quad_is_plausible(quad, W, H):
    """Reject segmentation failures: convex, near-rectangular angles, not glued to the frame."""
    q = np.asarray(quad, float)
    if not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
        return False
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        ang = np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)))
        if not 60 <= ang <= 120:
            return False
    tol = 0.01 * max(W, H)
    on_border = ((q[:, 0] < tol) | (q[:, 0] > W - tol) | (q[:, 1] < tol) | (q[:, 1] > H - tol)).sum()
    return on_border <= 1


def quad_homography(quad, width=1600, margin=0.15):
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    aspect = h / max(w, 1)
    W = width
    Hh = int(round(width * aspect))
    mx, my = margin * W, margin * W
    dst = np.array([[mx, my], [W + mx, my], [W + mx, Hh + my], [mx, Hh + my]], np.float32)
    Hm = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return Hm, (int(W + 2 * mx), int(Hh + 2 * my))


def _proj_score(ink_small, angle):
    h, w = ink_small.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    r = cv2.warpAffine(ink_small, M, (w, h), flags=cv2.INTER_NEAREST)
    p = r.sum(1).astype(np.float64)
    return float(np.var(p))


def estimate_skew(ink: np.ndarray, max_angle=20.0, test_rot90=False):
    """Dominant text/staff direction via projection-profile variance (degrees, + = CCW).

    Also tests the 90-degree hypothesis. Returns (angle, rotated90: bool).
    """
    sc = 600 / max(ink.shape)
    small = cv2.resize(ink * 255, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    small = (small > 40).astype(np.float32)
    best = {}
    for rot90 in ((False, True) if test_rot90 else (False,)):
        im = np.ascontiguousarray(np.rot90(small)) if rot90 else small
        angs = np.arange(-max_angle, max_angle + 0.01, 1.0)
        sc_ = [_proj_score(im, a) for a in angs]
        a0 = angs[int(np.argmax(sc_))]
        fine = np.arange(a0 - 1, a0 + 1.01, 0.1)
        sf = [_proj_score(im, a) for a in fine]
        best[rot90] = (fine[int(np.argmax(sf))], max(sf))
    rot90 = test_rot90 and best[True][1] > 1.3 * best[False][1]
    return best[rot90][0], rot90


def rotation_homography(angle_deg, size, rot90=False):
    """3x3 matrix rotating an image of `size` (w,h) about its centre; returns (M, new_size)."""
    w, h = size
    M = np.eye(3)
    if rot90:  # rotate 90 deg clockwise: (x,y) -> (h-1-y, x)
        M = np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], float)
        w, h = h, w
    R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0), [0, 0, 1]])
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
    rc = R @ corners
    x0, y0 = rc[0].min(), rc[1].min()
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    nw, nh = int(np.ceil(rc[0].max() - x0)), int(np.ceil(rc[1].max() - y0))
    return T @ R @ M, (nw, nh)
