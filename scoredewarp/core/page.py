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


def page_mask(img: np.ndarray):
    """Full-resolution mask of the paper (largest bright, low-saturation region).

    Same segmentation as ``detect_page_quad`` but without the quadrilateral
    requirement, so curled pages with curved borders keep their outline.
    Returns None when the region is implausible (tiny, or the whole frame).
    """
    H, W = img.shape[:2]
    sc = 800 / max(H, W)
    small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    score = cv2.GaussianBlur(lab[..., 0].astype(np.float32) - 0.6 * hsv[..., 1].astype(np.float32), (0, 0), 3)
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(5, int(0.02 * max(small.shape[:2])))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[i, cv2.CC_STAT_AREA] / m.size
    if area < 0.15 or area > 0.98:
        return None
    comp = (lab_ == i).astype(np.uint8) * 255
    # fill holes (dark ink regions inside the page)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comp = np.zeros_like(comp)
    cv2.drawContours(comp, cnts, -1, 255, -1)
    return cv2.resize(comp, (W, H), interpolation=cv2.INTER_NEAREST)


SIDES = ("top", "right", "bottom", "left")


def edge_strength(img: np.ndarray, pts: np.ndarray, work=800, win=3):
    """Max colour-gradient magnitude (grey levels / px at ``work`` scale) near each point."""
    H, W = img.shape[:2]
    sc = work / max(H, W)
    small = cv2.GaussianBlur(cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    small = small.astype(np.float32) if small.ndim == 3 else small[..., None].astype(np.float32)
    g = np.zeros(small.shape[:2], np.float32)
    for c in range(small.shape[2]):
        gx = cv2.Sobel(small[..., c], cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(small[..., c], cv2.CV_32F, 0, 1, ksize=3) / 8
        g = np.maximum(g, np.hypot(gx, gy))
    g = cv2.dilate(g, np.ones((2 * win + 1, 2 * win + 1), np.uint8))
    q = np.clip(np.round(np.asarray(pts) * sc).astype(int), 0, [g.shape[1] - 1, g.shape[0] - 1])
    return g[q[:, 1], q[:, 0]]


def page_edges(mask: np.ndarray, H_coarse: np.ndarray, *, img=None, min_grad=8.0, trim=0.06, border=0.006,
               step=20.0, min_span=0.25, max_bow=0.2):
    """Split the paper outline into its four sides, in coarse-frame coordinates.

    The outline of ``page_mask`` is mapped into the coarse (homography + deskew)
    frame, where the sheet is roughly axis aligned; the four corners are the
    extreme points along the diagonals and the outline between consecutive
    corners is one side. Each side is kept only where it is a real paper edge:
    corner rounding is trimmed, stretches lying on the photo frame (sheet cut
    off by the frame) are dropped, and a side must still span ``min_span`` of
    its length and bow by less than ``max_bow`` of it.

    With ``img``, only outline points on a sharp step (colour gradient >=
    ``min_grad``) count: a mask boundary through flat paper or background, or
    along a soft shadow, is not a paper edge.

    Returns {side: (N, 2) points in the coarse frame} for the usable sides.
    """
    if mask is None:
        return {}
    H0, W0 = mask.shape[:2]
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return {}
    c0 = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(c0) < 100:
        return {}
    # smooth the pixel staircase (mask is upsampled from ~800 px)
    k = max(3, int(round(max(H0, W0) / 800 * 3)) | 1)
    pad = np.concatenate([c0[-k:], c0, c0[:k]])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    c0 = np.stack([np.convolve(pad[:, d], ker, "same")[k:-k] for d in (0, 1)], 1)
    tol = border * max(H0, W0)
    on_frame = (c0[:, 0] < tol) | (c0[:, 0] > W0 - 1 - tol) | (c0[:, 1] < tol) | (c0[:, 1] > H0 - 1 - tol)
    if img is not None:
        on_frame |= edge_strength(img, c0) < min_grad
    c1 = cv2.perspectiveTransform(c0.reshape(-1, 1, 2), H_coarse).reshape(-1, 2)
    x, y = c1[:, 0], c1[:, 1]
    corner = {"tl": np.argmin(x + y), "tr": np.argmax(x - y), "br": np.argmax(x + y), "bl": np.argmax(y - x)}
    n = len(c1)
    order = sorted(corner, key=lambda kk: corner[kk])
    names = {frozenset(("tl", "tr")): "top", frozenset(("tr", "br")): "right",
             frozenset(("br", "bl")): "bottom", frozenset(("bl", "tl")): "left"}
    out = {}
    for a, b in zip(order, order[1:] + order[:1]):
        side = names.get(frozenset((a, b)))
        if side is None:
            return {}  # corners not in cyclic order: outline is not a sheet
        i0, i1 = corner[a], corner[b]
        idx = np.arange(i0, i1 + 1) if i1 >= i0 else np.concatenate([np.arange(i0, n), np.arange(0, i1 + 1)])
        if len(idx) < 20:
            continue
        pts, fr = c1[idx], on_frame[idx]
        seg = np.hypot(*np.diff(pts, axis=0).T)
        s = np.concatenate([[0], np.cumsum(seg)])
        L = s[-1]
        chord = np.linalg.norm(pts[-1] - pts[0])
        if L < 1 or chord < 0.5 * L:
            continue
        keep = (s > trim * L) & (s < (1 - trim) * L) & ~fr
        if keep.sum() < 10:
            continue
        sk = s[keep]
        if sk[-1] - sk[0] < min_span * L:
            continue
        # bow: distance from the chord through the corners
        d = pts[-1] - pts[0]
        nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
        if np.max(np.abs((pts[keep] - pts[0]) @ nrm)) > max_bow * chord:
            continue
        kp = pts[keep]
        sel, last = [0], sk[0]
        for i in range(1, len(sk)):
            if sk[i] - last >= step:
                sel.append(i)
                last = sk[i]
        out[side] = kp[sel]
    return out


def refine_page_mask(img: np.ndarray, rough: np.ndarray | None, structure_pts: np.ndarray,
                     work=480, iters=4, min_ratio=0.0):
    """GrabCut the sheet, seeded by what is known to be on it.

    Seeds: the convex hull of the detected score structures is certainly paper;
    the brightness mask ``rough`` is probably paper; a band along the photo frame
    is probably background. Colour models with several components take lit and
    shaded paper apart from the background, which a single brightness threshold
    cannot do under strong shading or a spotlight.
    Returns a full-resolution uint8 mask, or None.
    """
    H, W = img.shape[:2]
    pts = np.asarray(structure_pts, float)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 10:
        return rough
    sc = work / max(H, W)
    small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    if rough is not None:
        r = cv2.resize(rough, (w, h), interpolation=cv2.INTER_NEAREST)
        gc[r > 0] = cv2.GC_PR_FGD
    b = max(2, int(0.015 * max(h, w)))
    band = np.zeros((h, w), bool)
    band[:b], band[-b:], band[:, :b], band[:, -b:] = True, True, True, True
    gc[band & (gc == cv2.GC_PR_BGD)] = cv2.GC_BGD
    hull = cv2.convexHull(np.clip(pts * sc, 0, [w - 1, h - 1]).astype(np.int32))
    cv2.fillPoly(gc, [hull], cv2.GC_FGD)
    if not (gc == cv2.GC_BGD).any() and not (gc == cv2.GC_PR_BGD).any():
        return rough
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(0)   # GrabCut's GMM init is random: same photo, same sheet
        cv2.grabCut(small, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return rough
    m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    # optional: additions beyond the brightness mask must still look like paper
    # (off by default: it also drops strongly shadowed paper under a spotlight)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2)
    ref = gc == cv2.GC_FGD
    if rough is not None:
        ref = ref | (r > 0)
    paper_lv = np.percentile(gray[ref], 75) if ref.any() else 255.0
    added = (m > 0) & (gc != cv2.GC_FGD) & ((r == 0) if rough is not None else True)
    if min_ratio > 0:
        m[added & (gray < min_ratio * paper_lv)] = 0
    k = max(3, int(0.01 * max(h, w))) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return rough
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == i).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comp = np.zeros_like(comp)
    cv2.drawContours(comp, cnts, -1, 255, -1)
    if comp.mean() / 255 > 0.98:
        return None
    up = cv2.resize(comp, (W, H), interpolation=cv2.INTER_LINEAR)
    return (up > 127).astype(np.uint8) * 255
