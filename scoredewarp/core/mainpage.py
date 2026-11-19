"""Main page only: crop the rectified output to the page in front.

A hand-held photo of an open book keeps, around the flattened page, a strip of
the facing page, the paper rim past the last system (bent where the page
curls at the spine) and background. The geometry is not changed: the output
canvas (and its backward map) is cropped to the printed block of the main page
plus the content belonging to it (page number, title, footnotes, systems the
final detection missed), with a white margin, and background inside that box
is whitened.

Block: the final detection on the output (staves, jianpu rows; long text rows
only when there are neither). A facing page shows there as a column of
structures beyond a gutter (``pagenet.gutter``: a line crossed by no
structure); it is cut off only when it holds a small part of all structure
length, so that two balanced columns of one page are kept.

Print outside the block (``content_blobs``) widens the box when it lies
within the block's width (+- 3 %) and half its height, and either passes the
strict test (mostly inside that region, not at the photo frame, text-sized or
a staff row as wide as half the block, not solid dark, partly on the sheet) or,
with a sheet mask, is print on the sheet: mostly on it, paper-bright around
its ink, not solid dark.

Whitening needs a sheet mask (``--models page``, else the brightness mask):
off the sheet, and outside the block's rim also areas far darker than the
paper (a chair or table the learned mask leaked onto, a dark band past the
sheet edge) -- but never print on the sheet, and only background reaching in
from the box border: an island inside the page is print.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .pagenet import gutter
from .preprocess import content_blobs
from .structures import crop_structures


def block_items(S):
    """(x0, x1, y0, y1, weight) of the score structures on the output canvas."""
    it = []
    for st in S.staves:
        if len(st.tuples):
            t = np.asarray(st.tuples, float)
            it.append((st.x_range[0], st.x_range[1], t[:, 0, 1].min(), t[:, 4, 1].max(),
                       st.x_range[1] - st.x_range[0]))
    rows = [r for r in S.rows if r.kind == "jianpu"]
    if not it and not rows:
        rows = [r for r in S.rows if len(r.blobs) >= 5]
    for r in rows:
        if r.blobs:
            x0, x1 = min(b.x0 for b in r.blobs), max(b.x1 for b in r.blobs)
            it.append((x0, x1, min(b.y0 for b in r.blobs), max(b.y1 for b in r.blobs), x1 - x0))
    return np.array(it, float).reshape(-1, 5)


def main_page_box(image, S, paper=None, *, min_gap=0.03, minor=0.3, pad=0.03, look=0.5, margin=0.03,
                  max_text=8.0, rim=4.0, density=0.08, inside=0.75, dark=0.5, solid=0.4, paper_bg=0.8, work=1600):
    """Crop box (x0, y0, x1, y1) of the main page on the output, the gutter
    line (c, k, keep_left: x = c + k * y) of a cut-off facing page, the page
    mask (uint8, output size: sheet, block and the print accepted for it; None
    without a usable ``paper``) and info; box None when there is no reliable
    block."""
    H, W = image.shape[:2]
    info = {}
    it = block_items(S)
    if len(it) < 2:
        return None, None, None, {"skipped": "few structures"}
    keep = np.ones(len(it), bool)
    line = None
    yc = 0.5 * (it[:, 2] + it[:, 3])
    cut = gutter(np.c_[it[:, 0], it[:, 1], yc, it[:, 4]], W, min_gap=min_gap, min_side=0.0)
    if cut is not None:
        c, a, keep_left = cut
        k = np.tan(a)
        left = 0.5 * (it[:, 0] + it[:, 1]) - k * yc < c
        side = left if keep_left else ~left
        share = float(it[~side, 4].sum() / it[:, 4].sum())
        if share <= minor:
            keep, line = side, (float(c), float(k), bool(keep_left))
            info["gutter"] = {"x": round(float(c), 1), "deg": round(float(np.degrees(a)), 1),
                              "keep": "left" if keep_left else "right", "dropped_share": round(share, 3)}
        else:
            info["two_columns"] = round(share, 3)
    # measure numbers, ledger-line notes, dynamics: within a few spacings of the staves
    sp0 = S.spacing_est or max(10.0, min(H, W) / 120)
    bx0, bx1 = it[keep, 0].min(), it[keep, 1].max()
    by0, by1 = max(0.0, it[keep, 2].min() - rim * sp0), min(H - 1.0, it[keep, 3].max() + rim * sp0)
    bw, bh = bx1 - bx0, by1 - by0
    # search region for the page's own print outside the block, at work scale
    f = min(1.0, work / max(H, W))
    sp = sp0 * f
    img = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else image
    h, w = img.shape[:2]
    rx0, rx1 = max(0, int((bx0 - pad * bw) * f)), min(w, int(np.ceil((bx1 + pad * bw) * f)))
    ry0, ry1 = max(0, int((by0 - look * bh) * f)), min(h, int(np.ceil((by1 + look * bh) * f)))
    region = np.zeros((h, w), np.uint8)
    region[ry0:ry1, rx0:rx1] = 255
    if line is not None:
        c, k, keep_left = line
        yy, xx = np.mgrid[0:h, 0:w]
        far = (xx / f - k * yy / f >= c) if keep_left else (xx / f - k * yy / f < c)
        region[far] = 0
    block = (int(bx0 * f), int(by0 * f), int(np.ceil(bx1 * f)), int(np.ceil(by1 * f)))
    # the staves / rows themselves (no rim): always sheet, whatever the mask says
    core = (block[0], int((by0 + rim * sp0) * f), block[2], int(np.ceil((by1 - rim * sp0) * f)))
    # paper brightness: blurred so that thin ink vanishes, solid background stays dark
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.GaussianBlur(gray, (0, 0), max(2.0, sp)).astype(np.float32)
    g2 = cv2.GaussianBlur(gray, (0, 0), max(1.0, 0.3 * sp)).astype(np.float32)   # fine scale
    paper_lv = np.percentile(g[block[1]:block[3], block[0]:block[2]], 75)
    dark_lv = dark * paper_lv
    # a blob this dark on average is a solid object (chair, table edge); dense
    # print (a run of beamed chords: ~0.6 of the paper) is lighter
    solid_lv = solid * paper_lv
    page = sheet = None
    if paper is not None:
        pm = (cv2.resize(paper, (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        bb = pm[block[1]:block[3], block[0]:block[2]]
        if bb.size and bb.mean() >= 0.9:   # a mask that holds the block: print off it is background
            r = max(3, int(round(2 * sp))) | 1
            region[cv2.dilate(pm, np.ones((r, r), np.uint8)) == 0] = 0
            # slightly inside the sheet (its dark rim goes); block and accepted
            # print are added back below. The learned mask may hold dark
            # background beside a curled edge (IMG_3686: a chair under the
            # page): large areas far darker than the paper are not sheet
            r = max(3, int(round(0.5 * sp))) | 1
            page = cv2.erode(pm, np.ones((r, r), np.uint8)) * 255
            sheet = pm > 0
            # Not within the block's rim: paper in shadow under the last system
            # is darker too (IMG_3669), and print there must stay.
            far = np.ones((h, w), bool)
            far[block[1]:block[3], block[0]:block[2]] = False
            page[(g < dark_lv) & far] = 0
            # and thin dark bands at the sheet border, which the wide blur mixes
            # with the white beyond the photo: dark at fine scale and at least
            # ~0.6 spacing thick (staff lines, stems and beams are thinner; a
            # notehead or chord is an island or protected print, not whitened)
            k = max(3, int(round(0.6 * sp))) | 1
            band = cv2.morphologyEx((g2 < dark_lv).astype(np.uint8), cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            page[(band > 0) & far] = 0
            page[core[1]:core[3], core[0]:core[2]] = 255
            info["paper"] = True
    region[block[1]:block[3], block[0]:block[2]] = 255
    # small print (page / measure numbers) is sparser than a staff: lower
    # density than the protect default, still above background speckle (~4 %)
    # (``inside``: mostly within the region -- background texture spills over its edge)
    blobs, _ = content_blobs(img, region, sp, min_inside=inside, min_density=density, min_thick=1.0, min_size=1.0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    X0, Y0, X1, Y1 = block
    added = 0
    for i in range(1, n):
        x, y, bw_, bh_ = stats[i, :4]
        on_block = x < block[2] and x + bw_ > block[0] and y < block[3] and y + bh_ > block[1]
        if not on_block:
            if x <= 1 or y <= 1 or x + bw_ >= w - 1 or y + bh_ >= h - 1:
                continue
            # taller than text: only a staff row the detection missed (a small
            # violin staff above the piano), as wide as half the block
            if bh_ > max_text * sp and bw_ < 0.5 * (block[2] - block[0]):
                continue
            sub = lab[y:y + bh_, x:x + bw_] == i
            if g2[y:y + bh_, x:x + bw_][sub].mean() < solid_lv:
                continue   # solid dark object (a chair leg), not print on paper
            if sheet is not None and sheet[y:y + bh_, x:x + bw_][sub].mean() < 0.3:
                continue   # beyond the sheet (a dark band past its edge); print cut by the mask is partly on it
            added += 1
        X0, Y0, X1, Y1 = min(X0, x), min(Y0, y), max(X1, x + bw_), max(Y1, y + bh_)
        if page is not None:   # print the mask cuts (a page number in a curled corner)
            sub = lab[y:y + bh_, x:x + bw_] == i
            page[y:y + bh_, x:x + bw_][sub] = 255
    info["blobs_added"] = added
    if page is not None:
        # never whiten print on the sheet: every print-like blob on it
        # (protect_content defaults) that is not a solid dark object stays,
        # accepted for the box or not -- a staff cut by the photo frame, a
        # dense chord the dark-band test would take. Such print within the
        # search region also widens the box: a system the final detection
        # missed (its blob may merge with a curled facing page's print, so
        # the strict acceptance above rejects it)
        # Print = mostly on the sheet, around its ink paper-bright (not a
        # chair or table the learned mask leaked onto), not solid dark.
        pb, pink = content_blobs(img, sheet.astype(np.uint8) * 255, sp)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(pb, connectivity=8)
        if n2 > 1:
            flat = lab2.ravel()
            area = np.bincount(flat, minlength=n2).astype(float)
            mean = np.bincount(flat, weights=g2.ravel(), minlength=n2) / np.maximum(area, 1)
            in_reg = np.bincount(flat, weights=(region.ravel() > 0), minlength=n2) / np.maximum(area, 1)
            on = np.bincount(flat, weights=sheet.ravel(), minlength=n2) / np.maximum(area, 1)
            gap = pink.ravel() == 0
            n_gap = np.bincount(flat[gap], minlength=n2)
            bg = np.bincount(flat[gap], weights=gray.ravel()[gap].astype(float), minlength=n2) / np.maximum(n_gap, 1)
            ok = (mean >= solid_lv) & (on >= 0.75) & (bg >= paper_bg * paper_lv)
            ok[0] = False
            page[ok[lab2]] = 255
            info["protected"] = int(ok.sum())
            grow = ok & (in_reg >= 0.25) & (area >= (2 * sp) ** 2)
            for i in np.nonzero(grow)[0]:
                x, y, bw_, bh_ = st2[i, :4]
                X0, Y0, X1, Y1 = min(X0, x), min(Y0, y), max(X1, x + bw_), max(Y1, y + bh_)
        page = cv2.resize(page, (W, H), interpolation=cv2.INTER_NEAREST)
    # blobs may run into background or the facing page: stay in the search region
    X0, Y0, X1, Y1 = max(X0, rx0), max(Y0, ry0), min(X1, rx1), min(Y1, ry1)
    m = margin * (X1 - X0)
    box = (max(0, int((X0 - m) / f)), max(0, int((Y0 - m) / f)),
           min(W, int(np.ceil((X1 + m) / f))), min(H, int(np.ceil((Y1 + m) / f))))
    return box, line, page, info


def main_page(res, sheet=None, **kw):
    """``Result`` cropped to the main page (see module doc); unchanged when no
    block is found. ``res.stages["main_page"]`` records the decision.
    ``sheet``: learned sheet mask on the photo (``pagenet.sheet_mask``); it
    stands for the paper in place of ``res.paper``, a brightness mask that also
    holds bright background (lace, a white table)."""
    H, W = res.image.shape[:2]
    paper = res.paper
    if sheet is not None:
        paper = (res.bmap.apply(sheet, interp=cv2.INTER_LINEAR, value=0) > 127).astype(np.uint8) * 255
    box, line, page, info = main_page_box(res.image, res.structures, paper, **kw)
    res.stages["main_page"] = info
    if box is None:
        return res
    x0, y0, x1, y1 = box
    info["box"] = [x0, y0, x1, y1]
    info["kept_area"] = round((x1 - x0) * (y1 - y0) / float(W * H), 3)
    if (x0, y0, x1, y1) == (0, 0, W, H) and line is None and page is None:
        return res
    image = res.image[y0:y1, x0:x1].copy()
    enhanced = res.enhanced[y0:y1, x0:x1].copy() if res.enhanced.shape[:2] == (H, W) else res.enhanced
    if page is not None:
        paper = page[y0:y1, x0:x1].copy()
    elif res.paper is not None and res.paper.shape[:2] == (H, W):
        paper = res.paper[y0:y1, x0:x1].copy()
    else:
        paper = None
    if line is not None:   # facing page reaching into the box beyond the gutter
        c, k, keep_left = line
        yy, xx = np.mgrid[y0:y1, x0:x1]
        far = (xx - k * yy >= c) if keep_left else (xx - k * yy < c)
        image[far] = 255
        if enhanced.shape[:2] == far.shape:
            enhanced[far] = 255
        if paper is not None:
            paper[far] = 0
    content = None
    if res.content is not None and len(res.content):
        P = np.asarray(res.content, float) - [x0, y0]
        ok = (P[:, 0] >= 0) & (P[:, 0] < x1 - x0) & (P[:, 1] >= 0) & (P[:, 1] < y1 - y0)
        if line is not None:
            q = P + [x0, y0]
            ok &= (q[:, 0] - k * q[:, 1] < c) if keep_left else (q[:, 0] - k * q[:, 1] >= c)
        content = P[ok] if ok.sum() >= 3 else None
    if page is not None:
        # background inside the box (paper rim bent at the spine, lace, a
        # table) white: off the sheet, the block and the page's print -- and
        # only where it reaches in from the box border. Background surrounds
        # the page; an island inside it (a dense chord a darkness test took)
        # is print and stays
        off = (paper == 0).astype(np.uint8)
        n, lab = cv2.connectedComponents(off, connectivity=8)
        if n > 1:
            edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
            reach = np.zeros(n, bool)
            reach[edge] = True
            reach[0] = False
            off = reach[lab]
            paper[(paper == 0) & ~off] = 255
        else:
            off = off > 0
        image[off] = 255
        if enhanced.shape[:2] == off.shape:
            enhanced[off] = 255
        info["whitened"] = round(float(off.mean()), 3)
    bm = res.bmap.pad(-x0, -y0, -(W - x1), -(H - y1))
    return replace(res, image=image, enhanced=enhanced, bmap=bm, paper=paper, content=content,
                   structures=crop_structures(res.structures, x0, y0, x1, y1))
