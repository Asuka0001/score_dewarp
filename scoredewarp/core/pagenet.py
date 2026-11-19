"""Optional learned sheet segmentation (detection only, no warping).

Scanner apps find the sheet with a small segmentation network first and read
content only on it. Without that, off-sheet clutter -- wood grain, brick,
cloth folds, a hand, the facing page of a book -- turns into free horizontal
lines, vertical strokes and text rows that the surface fit can only reject
through its robust loss.

Model: U²-Netp (salient object segmentation, 1.1 M parameters, Apache-2.0,
``models/u2netp``; see its NOTICE.md). It was trained on generic salient
objects, not on documents, and saliency is not a sheet detector: when the page
fills the frame it marks the staves themselves (CPMS close-ups). ``sheet_mask``
therefore returns a mask only when it is sheet-like and holds the score that
was detected without it; otherwise the caller keeps the unfiltered evidence.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = _REPO / "models" / "u2netp" / "u2netp.onnx"
_SESSION = None


def model_path() -> Path:
    return Path(os.environ.get("SCOREDEWARP_PAGENET", DEFAULT_PATH))


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return model_path().exists()


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        p = model_path()
        if not p.exists():
            raise FileNotFoundError(f"page segmentation model not found: {p}")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 1
        _SESSION = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
    return _SESSION


def saliency(img: np.ndarray, size=320) -> np.ndarray:
    """BGR photo -> float32 (size, size) foreground map in [0, 1] (min-max
    normalised as in U²-Net's own inference)."""
    s = _session()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    p = s.run(None, {s.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    return ((p - p.min()) / max(float(p.max() - p.min()), 1e-9)).astype(np.float32)


def page_quad(img: np.ndarray, sal: np.ndarray | None = None, *, thr=0.5, min_area=0.15, max_area=0.97,
              solidity=0.9, min_iou=0.85, work=800):
    """Page quadrilateral (tl, tr, br, bl) in photo pixels from the saliency
    mask, or None: the largest component must be solid and page-sized, and a
    4-gon of its hull must explain it (IoU >= ``min_iou``), as for the
    brightness segmentation in ``page.detect_page_quad`` (corner angles 60-120
    deg, not glued to the frame). A full-frame page gives no quad (stripes)."""
    from .page import order_quad, quad_is_plausible
    H, W = img.shape[:2]
    f = work / max(H, W)
    w, h = max(1, int(round(W * f))), max(1, int(round(H * f)))
    p = cv2.resize(saliency(img) if sal is None else sal, (w, h), interpolation=cv2.INTER_LINEAR)
    m = (p > thr).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    area = cv2.contourArea(c) / (w * h)
    if not min_area <= area <= max_area or cv2.contourArea(c) / max(cv2.contourArea(hull), 1e-9) < solidity:
        return None
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.12, 23):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
    q = order_quad(quad)
    comp = np.zeros((h, w), np.uint8)
    cv2.drawContours(comp, [c], -1, 1, -1)
    qm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(qm, [q.astype(np.int32)], 1)
    if (comp & qm).sum() / max((comp | qm).sum(), 1) < min_iou:
        return None
    q = q / f
    if not quad_is_plausible(q, W, H):
        return None
    return q if edge_support(img, q) >= 3 else None


def edge_support(img, quad, offset=0.015, step=15.0, frac=0.6, n=40):
    """Number of quad sides that run along a paper border: at ``n`` points per
    side the grey levels ``offset`` (fraction of the photo size) inside and
    outside the quad differ by more than ``step`` (either way: shaded paper may
    be darker than a lit background) at a share of at least ``frac`` of the
    points. A sheet filling the frame gives a saliency quad through the paper
    itself (no step) -- not a page outline."""
    g = cv2.GaussianBlur(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.float32)
    H, W = g.shape
    d = offset * max(H, W)
    c = quad.mean(0)
    sides = 0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        t = np.linspace(0.1, 0.9, n)[:, None]
        p = a + t * (b - a)
        nrm = np.array([-(b - a)[1], (b - a)[0]], float)
        nrm /= max(np.linalg.norm(nrm), 1e-9)
        if np.dot(c - (a + b) / 2, nrm) < 0:
            nrm = -nrm                                  # points inward
        pin, pout = p + d * nrm, p - d * nrm
        ok = ((pout[:, 0] >= 0) & (pout[:, 0] < W) & (pout[:, 1] >= 0) & (pout[:, 1] < H))
        if ok.sum() < n // 2:
            continue
        vi = cv2.remap(g, pin[:, :1].astype(np.float32), pin[:, 1:].astype(np.float32), cv2.INTER_LINEAR)
        vo = cv2.remap(g, pout[:, :1].astype(np.float32), pout[:, 1:].astype(np.float32), cv2.INTER_LINEAR)
        sides += float((np.abs(vi - vo).ravel()[ok] > step).mean()) >= frac
    return sides


def sheet_mask(img: np.ndarray, content_pts: np.ndarray | None, *, sal=None, thr=0.5, hold=0.97, min_area=0.08,
               max_area=0.97, solidity=0.85, work=640):
    """Sheet mask on the photo (uint8 0/255, full resolution) or None, and a
    diagnostics dict.

    ``content_pts``: (N, 2) photo points of score structures detected without
    the mask (staff lines, jianpu rows). The component of the thresholded
    saliency holding most of them is taken, its holes filled (shaded paper,
    dark notation). Rejected unless it covers between ``min_area`` and
    ``max_area`` of the photo (a sheet filling the frame leaves no clutter to
    remove), is solid (area / convex hull area >= ``solidity``: saliency on a
    full-frame page marks the staves as stripes) and holds at least ``hold``
    of the content points."""
    H, W = img.shape[:2]
    info = {}
    if content_pts is None or len(content_pts) < 20:
        info["rejected"] = "no content"
        return None, info
    f = work / max(H, W)
    w, h = max(1, int(round(W * f))), max(1, int(round(H * f)))
    p = cv2.resize(saliency(img) if sal is None else sal, (w, h), interpolation=cv2.INTER_LINEAR)
    m = (p > thr).astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    q = np.round(np.asarray(content_pts, float) * f).astype(int)
    ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    q = q[ok]
    if n < 2 or not len(q):
        info["rejected"] = "empty"
        return None, info
    votes = np.bincount(lab[q[:, 1], q[:, 0]], minlength=n)
    votes[0] = 0
    k = int(votes.argmax())
    comp = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, [c], -1, 1, -1)
    area = float(filled.mean())
    hull = cv2.contourArea(cv2.convexHull(c))
    sol = float(cv2.contourArea(c) / max(hull, 1e-9))
    # tolerance of ~1 % of the photo for points right at the border
    tol = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.01 * work) + 1,) * 2))
    held = float(tol[q[:, 1], q[:, 0]].mean())
    info.update(area=round(area, 4), solidity=round(sol, 4), holds=round(held, 4))
    if not min_area <= area <= max_area:
        info["rejected"] = "area"
    elif sol < solidity:
        info["rejected"] = "solidity"
    elif held < hold:
        info["rejected"] = "holds"
    if "rejected" in info:
        return None, info
    full = cv2.resize(filled * 255, (W, H), interpolation=cv2.INTER_LINEAR)
    return (full > 127).astype(np.uint8) * 255, info


def content_points(S, H_coarse) -> np.ndarray | None:
    """Photo points of the score structures in ``S`` (detected on the coarse
    canvas of ``H_coarse``): staff tuples and jianpu rows; text rows only when
    there are neither (background clutter forms text-like rows too)."""
    P = [np.asarray(t, float).reshape(-1, 2) for st in S.staves for t in st.tuples]
    for r in [r for r in S.rows if r.kind == "jianpu"]:
        P.append(np.array([[b.cx, b.cy] for b in r.blobs], float))
    if sum(len(p) for p in P) < 20:
        for r in [r for r in S.rows if r.kind == "text" and len(r.blobs) >= 5]:
            P.append(np.array([[b.cx, b.cy] for b in r.blobs], float))
    if not P:
        return None
    P = np.concatenate(P)
    return cv2.perspectiveTransform(P.reshape(-1, 1, 2), np.linalg.inv(H_coarse)).reshape(-1, 2)


def _items(S):
    """(x0, x1, y, weight) on the detection canvas: staves, else jianpu rows,
    else long text rows."""
    it = [(st.x_range[0], st.x_range[1], float(np.median([t[2, 1] for t in st.tuples])), st.x_range[1] - st.x_range[0])
          for st in S.staves if len(st.tuples)]
    if not it:
        rows = [r for r in S.rows if r.kind == "jianpu"] or [r for r in S.rows if len(r.blobs) >= 5]
        for r in rows:
            x0, x1 = min(b.x0 for b in r.blobs), max(b.x1 for b in r.blobs)
            it.append((x0, x1, float(np.median([b.cy for b in r.blobs])), x1 - x0))
    return np.array(it, float).reshape(-1, 4)


def spread_cut(S, *, min_gap=0.03, min_side=0.15, tilts=np.radians(np.arange(-12, 12.1, 1.0))):
    """Gutter of a two-page spread on the detection canvas, or None.

    A sheet mask of an open book often holds part of the facing page, whose
    staves sit on a different surface; one cylinder fitted to both pages
    squashes the main page. Staves (or rows) of a spread form two columns: a
    line x = c + tan(a) * y (|a| <= 12 deg) crossed by no structure, with at
    least ``min_side`` of the structure length on each side and a gap of at
    least ``min_gap`` of the canvas width. Returns (c, a, keep_left) with the
    side holding more structure length kept."""
    it = _items(S)
    if len(it) < 3:
        return None
    return gutter(it, S.shape[1], min_gap=min_gap, min_side=min_side, tilts=tilts)


def gutter(it, W, *, min_gap=0.03, min_side=0.15, tilts=np.radians(np.arange(-12, 12.1, 1.0))):
    """``spread_cut`` on items (x0, x1, y, weight) of a canvas of width ``W``."""
    tot = it[:, 3].sum()
    best = None
    for a in tilts:
        k = np.tan(a)
        x0, x1 = it[:, 0] - k * it[:, 2], it[:, 1] - k * it[:, 2]
        o = np.argsort(x0)
        reach = np.maximum.accumulate(x1[o])        # rightmost end of all items starting before
        for i in range(len(o) - 1):
            gap = x0[o[i + 1]] - reach[i]
            if gap < min_gap * W:
                continue
            left = it[o[:i + 1], 3].sum()
            if min(left, tot - left) < min_side * tot:
                continue
            if best is None or gap > best[0]:
                best = (gap, reach[i] + gap / 2, a, left >= tot - left)
    return None if best is None else best[1:]


def cut_mask(shape, cut, H_coarse, canvas_shape):
    """uint8 photo mask (255 = kept side of the gutter ``cut``)."""
    c, a, keep_left = cut
    Hc_, Wc_ = canvas_shape
    ys = np.linspace(-2 * Hc_, 3 * Hc_, 64)
    line = np.stack([c + np.tan(a) * ys, ys], 1)
    far = (-4 * Wc_) if keep_left else (5 * Wc_)
    poly = np.concatenate([line, [[far, ys[-1]], [far, ys[0]]]])
    q = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), np.linalg.inv(H_coarse)).reshape(-1, 2)
    m = np.zeros(shape[:2], np.uint8)
    q = np.clip(q, -1e5, 1e5)
    cv2.fillPoly(m, [np.round(q).astype(np.int32)], 255)
    return m


def outline_points(sheet: np.ndarray, n=400, frame=0.01):
    """Up to ``n`` photo points along the sheet mask's outer contour, without
    the stretches where the mask runs along the photo frame (the sheet is cut
    there, not bordered)."""
    H, W = sheet.shape[:2]
    cnts, _ = cv2.findContours((sheet > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    tol = frame * max(H, W)
    c = c[(c[:, 0] > tol) & (c[:, 0] < W - 1 - tol) & (c[:, 1] > tol) & (c[:, 1] < H - 1 - tol)]
    if len(c) < 40:
        return None
    return c[np.linspace(0, len(c) - 1, min(n, len(c))).astype(int)]
