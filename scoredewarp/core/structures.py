"""Structure detection on a coarsely rectified page.

Produces the geometric evidence used by the dewarping models:

* horizontal fragments  — traced line pieces (staff lines, ledger lines,
  jianpu 减时线/underlines). "Same flat Y along the fragment."
* staves                — fragments grouped into 5-line systems with equal
  spacing (五线谱 prior).
* text rows             — chains of character blobs; classified as jianpu note
  rows (digits 0-7) or generic text/lyrics rows (简谱 / 文本 prior:
  common baseline & top line, uniform glyph height, upright glyphs).
* vertical segments     — barlines, stems, upright strokes. "Same flat X."
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..synth.fonts import digit_font_paths, ink_bbox, load_font


# ---------------------------------------------------------------------------
@dataclass
class Fragment:
    pts: np.ndarray            # (N,2) sampled along the line
    staff: int = -1
    line: int = -1


@dataclass
class Staff:
    frag_ids: list
    tuples: list               # list of (5,2) arrays: points on line 0..4 at a common x
    spacing: float
    top_poly: np.ndarray = None  # polyfit coeffs of line 0, line 4 (y = f(x))
    bot_poly: np.ndarray = None
    x_range: tuple = (0, 0)


@dataclass
class Blob:
    x0: float
    y0: float
    x1: float
    y1: float
    digit: str | None = None

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2


@dataclass
class TextRow:
    blobs: list
    kind: str                  # 'jianpu' | 'text'
    height: float              # median glyph height
    full: list = field(default_factory=list)  # indices of full-height blobs


@dataclass
class VSeg:
    pts: np.ndarray            # (N,2) sampled along the segment


@dataclass
class Structures:
    shape: tuple
    thickness: float
    spacing_est: float
    fragments: list
    staves: list
    rows: list
    vsegs: list
    page_type: str = "unknown"
    regions: list = field(default_factory=list)   # (type, polygon (4,2))
    heads: np.ndarray = None   # (N,4) filled noteheads: cx, cy, width (moment-based), staff index

    def summary(self):
        return {
            "page_type": self.page_type,
            "n_staves": len(self.staves),
            "n_jianpu_rows": sum(r.kind == "jianpu" for r in self.rows),
            "n_text_rows": sum(r.kind == "text" for r in self.rows),
            "n_fragments": len(self.fragments),
            "n_vsegs": len(self.vsegs),
        }


# ---------------------------------------------------------------------------
def run_metrics(ink: np.ndarray):
    """Classic OMR run-length statistics -> (line thickness, line distance)."""
    cols = ink[:, ::3].T.astype(np.int8)
    a = np.pad(cols, ((0, 0), (1, 1))).ravel()
    tr = np.flatnonzero(np.diff(a))
    runs = np.diff(tr)
    black = runs[0::2]
    white = runs[1::2]
    n = min(len(black), len(white))
    black_s = black[black <= 10]
    thick = float(np.bincount(black_s).argmax()) if black_s.size else 2.0
    dist = black[:n] + white[:n]
    dist = dist[(dist >= 6) & (dist <= 60)]
    if dist.size < 50:
        return max(thick, 1.0), 0.0
    hist = np.bincount(dist, minlength=61).astype(float)
    hist = np.convolve(hist, [0.25, 0.5, 0.25], "same")
    return max(thick, 1.0), float(hist.argmax())


def _line_kernel(length, angle_deg, horizontal=True):
    L = int(length)
    a = np.deg2rad(angle_deg)
    if horizontal:
        dy = int(round(np.tan(a) * L / 2))
        k = np.zeros((2 * abs(dy) + 1, L), np.uint8)
        cv2.line(k, (0, abs(dy) - dy), (L - 1, abs(dy) + dy), 1, 1)
    else:
        dx = int(round(np.tan(a) * L / 2))
        k = np.zeros((L, 2 * abs(dx) + 1), np.uint8)
        cv2.line(k, (abs(dx) - dx, 0), (abs(dx) + dx, L - 1), 1, 1)
    return k


def directional_open(ink, length, horizontal=True, angles=tuple(range(-16, 17, 2)), thicken=True):
    thick = cv2.dilate(ink, np.ones((3, 1) if horizontal else (1, 3), np.uint8)) if thicken else ink
    out = np.zeros_like(ink)
    for a in angles:
        out |= cv2.morphologyEx(thick, cv2.MORPH_OPEN, _line_kernel(length, a, horizontal))
    return out & thick


def _trace(mask_lab, lab, bbox, horizontal, max_w, step, min_len):
    """Trace a thin component as a polyline: centroid per column (or row)."""
    x, y, w, h = bbox
    sub = mask_lab[y:y + h, x:x + w] == lab
    ys, xs = np.nonzero(sub)
    if not horizontal:
        xs, ys = ys, xs  # trace along rows
        n = h
    else:
        n = w
    cnt = np.bincount(xs, minlength=n).astype(float)
    sm = np.bincount(xs, weights=ys, minlength=n)
    mn = np.full(n, 1e9)
    mx = np.full(n, -1e9)
    np.minimum.at(mn, xs, ys)
    np.maximum.at(mx, xs, ys)
    ok = (cnt > 0) & (cnt <= max_w) & ((mx - mn + 1) <= cnt + 1)
    c = np.where(ok, sm / np.maximum(cnt, 1), np.nan)
    pieces = []
    idx = np.arange(n)
    # split at invalid gaps
    valid = ~np.isnan(c)
    if not valid.any():
        return pieces
    breaks = np.flatnonzero(np.diff(valid.astype(int)) != 0) + 1
    for seg in np.split(idx, breaks):
        if not valid[seg[0]] or len(seg) < min_len:
            continue
        cs = c[seg]
        if len(seg) >= 5:
            cs = np.convolve(np.pad(cs, 2, mode="edge"), np.ones(5) / 5, "valid")
        samp = np.unique(np.concatenate([seg[::step], seg[-1:]]))
        vals = np.interp(samp, seg, cs)
        if horizontal:
            pts = np.stack([samp + x, vals + y], 1)
        else:
            pts = np.stack([vals + x, samp + y], 1)
        pieces.append(pts.astype(float))
    return pieces


def _column_runs(col, max_w):
    a = np.concatenate([[0], col.astype(np.int8), [0]])
    d = np.diff(a)
    st = np.flatnonzero(d == 1)
    en = np.flatnonzero(d == -1)
    ln = en - st
    ok = ln <= max_w
    return (st[ok] + en[ok] - 1) / 2.0


def track_lines(mask, max_w, min_len, step=3, max_slope=0.4):
    """Track thin horizontal curves through a line mask.

    Per sampled column: centres of thin vertical runs. Runs in neighbouring
    columns are linked by mutual nearest neighbour (|dy| <= step*max_slope+1.5).
    Robust to adjacent lines touching through symbols, unlike connected
    components. Returns list of (N,2) polylines (x, y).
    """
    H, W = mask.shape
    cols = list(range(0, W, step))
    cents = [_column_runs(mask[:, x], max_w) for x in cols]
    max_dy = step * max_slope + 1.5
    chain_of = [np.full(len(c), -1) for c in cents]
    chains = []
    for k, x in enumerate(cols):
        a = cents[k]
        for i in range(len(a)):
            if chain_of[k][i] < 0:
                chain_of[k][i] = len(chains)
                chains.append([(x, a[i])])
        if k + 1 >= len(cols):
            break
        b = cents[k + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        j = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
        if len(b) > 1:
            j = np.where(np.abs(b[j - 1] - a) < np.abs(b[j] - a), j - 1, j)
        i2 = np.clip(np.searchsorted(a, b), 1, len(a) - 1) if len(a) > 1 else np.zeros(len(b), int)
        if len(a) > 1:
            i2 = np.where(np.abs(a[i2 - 1] - b) < np.abs(a[i2] - b), i2 - 1, i2)
        for i in range(len(a)):
            jj = j[i]
            if i2[jj] == i and abs(b[jj] - a[i]) <= max_dy:
                cid = chain_of[k][i]
                chain_of[k + 1][jj] = cid
                chains[cid].append((cols[k + 1], b[jj]))
    out = []
    for c in chains:
        if len(c) * step < min_len:
            continue
        p = np.array(c, float)
        if len(p) >= 5:
            p[:, 1] = np.convolve(np.pad(p[:, 1], 2, mode="edge"), np.ones(5) / 5, "valid")
        out.append(p[::2] if len(p) > 3 else p)
    return out


def find_fragments(ink, thick, spacing, horizontal=True, length=None):
    L = int(length or max(25, 3 * spacing))
    m = directional_open(ink, L, horizontal)
    max_w = 2.5 * thick + 4
    if horizontal:
        out = track_lines(m, max_w, int(L * 0.8))
    else:
        out = [p[:, ::-1].copy() for p in track_lines(np.ascontiguousarray(m.T), max_w, int(L * 0.8))]
    return out, m


def _interp_frag(f, x):
    p = f.pts
    if x < p[0, 0] or x > p[-1, 0]:
        return None
    return float(np.interp(x, p[:, 0], p[:, 1]))


def group_staves(frags, W, spacing_hint, *, bridge_gaps=True):
    """5 equally spaced fragments at many columns -> staves (union-find)."""
    step = max(8, int(spacing_hint) if spacing_hint else 12)
    parent = list(range(len(frags)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tuples = []  # (x, ys(5), fids(5))
    for x in range(0, W, step):
        ys = []
        for fi, f in enumerate(frags):
            y = _interp_frag(f, x)
            if y is not None:
                ys.append((y, fi))
        ys.sort()
        i = 0
        while i + 4 < len(ys):
            yy = np.array([v[0] for v in ys[i:i + 5]])
            d = np.diff(yy)
            dm = np.median(d)
            ok = 5 <= dm <= 60 and np.all(np.abs(d - dm) < 0.22 * dm + 1.0)
            if ok and spacing_hint:
                ok = 0.55 * spacing_hint <= dm <= 1.8 * spacing_hint
            if ok:
                fids = [v[1] for v in ys[i:i + 5]]
                tuples.append((x, yy, fids))
                for a, b in zip(fids[:-1], fids[1:]):
                    parent[find(a)] = find(b)
                i += 5
            else:
                i += 1
    # Symbols and barlines interrupt all five tracks at once. Such sections
    # share no fragment IDs, so fragment union alone splits a genuine staff
    # into short groups that fail the page-span check below. Join neighbouring
    # five-line observations only when their spacing and vertical position
    # agree. Never invent line points inside the gap.
    recent = []
    for x, yy, fids in (tuples if bridge_gaps else ()):
        sp = float(np.median(np.diff(yy)))
        recent = [t for t in recent if x - t[0] <= 6 * step]
        candidates = []
        for px, py, pf in recent:
            if px == x:
                continue
            psp = float(np.median(np.diff(py)))
            if abs(sp - psp) > 0.2 * min(sp, psp) + 0.5:
                continue
            delta = np.abs(yy - py)
            limit = min(0.75 * min(sp, psp), 2 + 0.35 * (x - px))
            if delta.max() <= limit:
                candidates.append((x - px, float(delta.mean()), pf))
        if candidates:
            _, _, previous = min(candidates, key=lambda c: c[:2])
            parent[find(fids[0])] = find(previous[0])
        recent.append((x, yy, fids))
    groups = {}
    for x, yy, fids in tuples:
        groups.setdefault(find(fids[0]), []).append((x, yy, fids))
    staves = []
    for g, tl in groups.items():
        xs_t = [t[0] for t in tl]
        # a real staff spans a large part of the page width
        if len(tl) < 3 or (max(xs_t) - min(xs_t)) < 0.25 * W:
            continue
        votes = {}
        for x, yy, fids in tl:
            for li, fi in enumerate(fids):
                votes.setdefault(fi, []).append(li)
        fids = sorted(votes)
        sp = float(np.median([np.median(np.diff(yy)) for _, yy, _ in tl]))
        tup = []
        for x, yy, fl in tl:
            # sample along the local normal of line 0 for slight robustness
            tup.append(np.stack([np.full(5, float(x)), yy], 1))
        st = Staff(frag_ids=fids, tuples=tup, spacing=sp)
        xs = np.array([t[0, 0] for t in tup])
        y0 = np.array([t[0, 1] for t in tup])
        y4 = np.array([t[4, 1] for t in tup])
        deg = 2 if len(xs) >= 8 else 1
        st.top_poly = np.polyfit(xs, y0, deg)
        st.bot_poly = np.polyfit(xs, y4, deg)
        fx = np.concatenate([frags[i].pts[:, 0] for i in fids])
        st.x_range = (float(fx.min()), float(fx.max()))
        for fi in fids:
            frags[fi].staff = len(staves)
            frags[fi].line = int(np.bincount(votes[fi]).argmax())
        staves.append(st)
    staves.sort(key=lambda s: np.polyval(s.top_poly, np.mean(s.x_range)))
    for k, st in enumerate(staves):
        for fi in st.frag_ids:
            frags[fi].staff = k
    return staves


# ---------------------------------------------------------------------------
_TEMPLATES = None


def _digit_templates():
    global _TEMPLATES
    if _TEMPLATES is None:
        from PIL import Image, ImageDraw
        T = []
        for path in digit_font_paths():
            for ch in "01234567":
                font = load_font(path, 80)
                img = Image.new("L", (200, 200), 0)
                ImageDraw.Draw(img).text((100, 150), ch, font=font, fill=255, anchor="ms")
                a = np.asarray(img) > 127
                ys, xs = np.nonzero(a)
                crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
                asp = crop.shape[1] / crop.shape[0]
                t = cv2.resize(crop, (16, 24), interpolation=cv2.INTER_AREA)
                t = (t - t.mean()) / (t.std() + 1e-6)
                T.append((ch, asp, t))
        _TEMPLATES = T
    return _TEMPLATES


def classify_digit(crop, *, reject_boxes=True):
    h, w = crop.shape
    asp = w / max(h, 1)
    # Missing-font boxes in score titles resemble "0" after resizing. Their
    # rectangular outer contour plus enclosed hole distinguishes them from a
    # rounded digit without suppressing legitimate all-rest (0000) rows.
    contours, hierarchy = cv2.findContours(crop.astype(np.uint8), cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)
    if reject_boxes and hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0, i, 3] >= 0 or hierarchy[0, i, 2] < 0:
                continue
            rect = cv2.minAreaRect(contour)
            area = rect[1][0] * rect[1][1]
            approx = cv2.approxPolyDP(contour, .03 * cv2.arcLength(contour, True), True)
            if area > 0 and cv2.contourArea(contour) / area > .87 and len(approx) == 4:
                return None, -1.
    t = cv2.resize(crop.astype(np.float32), (16, 24), interpolation=cv2.INTER_AREA)
    t = (t - t.mean()) / (t.std() + 1e-6)
    best, bch = -1.0, None
    for ch, a, tm in _digit_templates():
        if not (0.65 * a <= asp <= 1.5 * a):
            continue
        s = float((t * tm).mean())
        if s > best:
            best, bch = s, ch
    return (bch, best) if best > 0.55 else (None, best)


def staff_band_mask(shape, staves, margin_sp=2.6):
    m = np.zeros(shape, np.uint8)
    for st in staves:
        xs = np.linspace(st.x_range[0] - st.spacing, st.x_range[1] + st.spacing, 40)
        top = np.polyval(st.top_poly, xs) - margin_sp * st.spacing
        bot = np.polyval(st.bot_poly, xs) + margin_sp * st.spacing
        poly = np.concatenate([np.stack([xs, top], 1), np.stack([xs[::-1], bot[::-1]], 1)])
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    return m


def find_noteheads(ink, staves, sp):
    """Filled noteheads: every one is the same printed glyph, so their flat
    widths are equal everywhere on the page -- a horizontal ruler (staff
    spacing is only a vertical one). An opening with a disk thicker than staff
    lines, stems and beams leaves the head bodies; chords / seconds merge into
    blobs of the wrong size and are dropped. Returns (N,4): cx, cy, width, staff."""
    if not staves or not sp:
        return np.zeros((0, 4))
    d = max(3, int(round(0.65 * sp)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    band = staff_band_mask(ink.shape, staves, margin_sp=1.5)
    op = cv2.morphologyEx((ink & band).astype(np.uint8), cv2.MORPH_OPEN, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(op, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (0.7 * sp <= h <= 1.5 * sp and 0.9 * sp <= w <= 2.0 * sp and a >= 0.6 * w * h):
            continue
        cx, cy = cent[i]
        dist = [abs(cy - _staff_mid(st, cx)) / st.spacing for st in staves]
        j = int(np.argmin(dist))
        if dist[j] > 4.0:
            continue
        xs = np.nonzero(lab[y:y + h, x:x + w] == i)[1]
        out.append((cx, cy, 4.0 * float(np.sqrt(xs.var() + 1 / 12)), j))   # full width of an ellipse = 4 sigma_x
    return np.array(out, float).reshape(-1, 4)


def _merge_chains(chains, B, max_gap=10.0, max_dy=0.9, max_ratio=1.6):
    """Join row pieces split by wide gaps (jianpu "5 - - |", rests, lyrics with
    blanks), dots or underlines: the longer piece's centre line, extrapolated
    to the other piece, must hit it within ``max_dy`` glyph heights (adjacent
    lyric rows are ~2.5 heights away), with similar glyph heights and a gap
    below ``max_gap`` heights. Greedy, best pair first.
    Like assembling text spans into lines (page_dewarp; Kim et al. 2015)."""
    cx = (B[:, 0] + B[:, 2]) / 2
    cy = (B[:, 1] + B[:, 3]) / 2
    hh = B[:, 3] - B[:, 1]
    chains = [sorted(c, key=lambda k: cx[k]) for c in chains]

    def line(c):
        h = float(np.median(hh[c]))
        if len(c) >= 4 and np.ptp(cx[c]) > 3 * h:
            # fit on the middle half of heights: dots / ascenders do not tilt it
            m = np.abs(hh[c] - h) <= 0.25 * h
            idx = np.array(c)[m] if m.sum() >= 3 else np.array(c)
            k, b = np.polyfit(cx[idx], cy[idx], 1)
            if abs(k) < 0.15:
                return k, b, h
        return 0.0, float(np.median(cy[c])), h

    info = [line(c) for c in chains]
    while len(chains) > 1:
        k = np.array([i[0] for i in info])
        b0 = np.array([i[1] for i in info])
        h1 = np.array([i[2] for i in info])
        n = np.array([len(c) for c in chains])
        last = np.array([c[-1] for c in chains])
        first = np.array([c[0] for c in chains])
        # a (rows) left of b (columns)
        H = np.maximum(h1[:, None], h1[None, :])
        ok = H <= max_ratio * np.minimum(h1[:, None], h1[None, :])
        gap = B[first, 0][None, :] - B[last, 2][:, None]
        xa, xb = cx[last][:, None], cx[first][None, :]
        ok &= (gap >= -0.3 * H) & (gap <= max_gap * H) & (xb > xa)
        np.fill_diagonal(ok, False)
        if not ok.any():
            break
        # both centre lines must agree at the junction
        d1 = np.abs(k[:, None] * xb + b0[:, None] - cy[first][None, :]) * (n[:, None] >= 2)
        d2 = np.abs(k[None, :] * xa + b0[None, :] - cy[last][:, None]) * (n[None, :] >= 2)
        d3 = np.abs(cy[last][:, None] - cy[first][None, :])
        # the longer piece's centre line is the reliable one; short pieces
        # (< 4 glyphs) carry no direction, so they are judged by that line only
        na, nb = n[:, None], n[None, :]
        dy = np.where(na >= nb, d1, d2)
        dy = np.where(np.maximum(na, nb) >= 4, dy, d3)
        ok &= dy <= max_dy * H
        if not ok.any():
            break
        score = np.where(ok, dy / H + 0.02 * gap / H, np.inf)
        a, b = np.unravel_index(np.argmin(score), score.shape)
        chains[a] = chains[a] + chains[b]
        info[a] = line(chains[a])
        del chains[b], info[b]
    return chains


def _chain_blobs(B, max_gap=5.0, max_ratio=2.0):
    """Chains of blobs (rows of ``B``: x0, y0, x1, y1) by mutual nearest right
    neighbours with vertical overlap; returns lists of row indices of ``B``."""
    hh = B[:, 3] - B[:, 1]
    cx = (B[:, 0] + B[:, 2]) / 2
    gap = B[None, :, 0] - B[:, None, 2]                         # x0_j - x1_i
    ov = np.minimum(B[:, None, 3], B[None, :, 3]) - np.maximum(B[:, None, 1], B[None, :, 1])
    mh = np.minimum(hh[:, None], hh[None, :])
    Mh = np.maximum(hh[:, None], hh[None, :])
    ok = (cx[None, :] > cx[:, None]) & (gap > -0.25 * mh) & (gap < max_gap * Mh) & (ov > 0.5 * mh) & (Mh < max_ratio * mh)
    dist = np.where(ok, cx[None, :] - cx[:, None], np.inf)
    right = np.where(np.isfinite(dist.min(1)), dist.argmin(1), -1)
    left = np.where(np.isfinite(dist.min(0)), dist.argmin(0), -1)
    nxt = {i: int(right[i]) for i in range(len(B)) if right[i] >= 0 and left[right[i]] == i}
    has_prev = set(nxt.values())
    chains = []
    for i in range(len(B)):
        if i in has_prev:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        chains.append(chain)
    return chains


def find_text_rows(ink, line_len, band, *, reject_boxes=True, merge_rows=False, digit_first=False,
                   classifier="template"):
    """Jianpu note rows and text rows from glyph blobs. ``digit_first``: note
    rows are chained from digit blobs alone before the other glyphs are
    chained into text rows. ``classifier``: crop -> (digit or None, score),
    default ``classify_digit`` (templates)."""
    # remove only pixels belonging to long, strictly horizontal runs (underlines,
    # 减时线, ledger/staff remnants) without eroding the glyphs touching them
    lines = directional_open(ink, line_len, True, angles=(-2, 0, 2), thicken=False)
    mask = ink & (1 - lines) & (1 - band)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    ar = stats[1:, cv2.CC_STAT_AREA]
    sel = (hs >= 10) & (hs <= 150) & (ws / np.maximum(hs, 1) > 0.25) & (ws / np.maximum(hs, 1) < 1.4) & (ar > 30)
    if sel.sum() < 5:
        return [], 0.0
    # area-weighted so that glyph bodies (not dots / fragments) define the mode
    hist = np.bincount(hs[sel] // 2, weights=ar[sel].astype(float), minlength=80).astype(float)
    hist = np.convolve(hist, [0.25, 0.5, 0.25], "same")
    h_mode = float(hist.argmax() * 2 + 1)
    blobs = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if h < 0.45 * h_mode or h > 3.2 * h_mode or w > 3 * h_mode or a < 10:
            continue
        if w / h < 0.12 and h > 1.4 * h_mode:  # barline-like
            continue
        blobs.append(Blob(float(x), float(y), float(x + w), float(y + h)))
        blobs[-1]._lab = i
    if len(blobs) < 3:
        return [], h_mode
    B = np.array([[b.x0, b.y0, b.x1, b.y1] for b in blobs])
    rows = []
    rest = np.arange(len(blobs))
    if digit_first:
        # jianpu note rows first, from digit blobs only: a lyric row of the
        # same glyph size right below otherwise captures the chain (nearest
        # right neighbour jumps between the two rows), the mixed row fails the
        # digit-fraction test and the page loses its jianpu prior
        # (dev 2026-11-19: 110 / 297 GT note rows not found as jianpu rows).
        crops = [lab[int(b.y0):int(b.y1), int(b.x0):int(b.x1)] == b._lab for b in blobs]
        if classifier == "cnn":
            from .glyphnet import classify
            res = classify(crops, thr=0.6)
        else:
            res = [classify_digit(c, reject_boxes=reject_boxes) for c in crops]
        for b, (d, _) in zip(blobs, res):
            b.digit = d
        dig = np.array([b.digit is not None for b in blobs])
        D = np.flatnonzero(dig)
        used = np.zeros(len(blobs), bool)
        if len(D) >= 3:
            ch = _merge_chains([list(D[c]) for c in _chain_blobs(B[D], max_gap=6.0, max_ratio=1.5)], B)
            for chain in ch:
                if len(chain) < 3:
                    continue
                rb = sorted([blobs[k] for k in chain], key=lambda b: b.cx)
                hmed = float(np.median([b.h for b in rb]))
                full = [k for k, b in enumerate(rb) if 0.8 * hmed <= b.h <= 1.25 * hmed]
                rows.append(TextRow(rb, "jianpu", hmed, full))
                used[chain] = True
        rest = np.flatnonzero(~used)
    chains = [list(rest[c]) for c in _chain_blobs(B[rest])] if len(rest) >= 3 else []
    if merge_rows and chains:
        chains = _merge_chains(chains, B)
    for chain in chains:
        if len(chain) < 3:
            continue
        rb = [blobs[k] for k in chain]
        hmed = float(np.median([b.h for b in rb]))
        n_dig = 0
        for b in rb:
            crop = (lab[int(b.y0):int(b.y1), int(b.x0):int(b.x1)] == b._lab)
            if 0.75 * hmed <= b.h <= 1.3 * hmed:
                b.digit, _ = classify_digit(crop, reject_boxes=reject_boxes)
                n_dig += b.digit is not None
        kind = "jianpu" if (n_dig >= 4 and n_dig >= 0.55 * len(rb)) else "text"
        full = [k for k, b in enumerate(rb) if 0.8 * hmed <= b.h <= 1.25 * hmed]
        rows.append(TextRow(rb, kind, hmed, full))
    rows.sort(key=lambda r: np.median([b.cy for b in r.blobs]))
    return rows, h_mode


def _stroke_runs(ink, p, reach=20):
    """Per image row along a traced vertical stroke: (y, left, right) of the
    ink run under the trace (NaN where the trace is on paper)."""
    H, W = ink.shape
    ys = np.arange(max(0, int(np.ceil(p[0, 1]))), min(H - 1, int(np.floor(p[-1, 1]))) + 1)
    xs = np.rint(np.interp(ys, p[:, 1], p[:, 0])).astype(int)
    out = np.full((len(ys), 3), np.nan)
    out[:, 0] = ys
    for i, (y, x) in enumerate(zip(ys, xs)):
        lo, hi = max(0, x - reach), min(W, x + reach + 1)
        row = ink[y, lo:hi] > 0
        c = x - lo
        hit = [c + d for d in (0, -1, 1, -2, 2) if 0 <= c + d < len(row) and row[c + d]]
        if not hit:
            continue
        c = hit[0]
        z = np.flatnonzero(~row[:c])
        a = z[-1] + 1 if len(z) else 0
        z = np.flatnonzero(~row[c:])
        b = c + z[0] - 1 if len(z) else len(row) - 1
        out[i, 1:] = a + lo, b + lo
    return out


def refine_strokes(segs, ink, min_span, tol=1.0, step=3):
    """Re-centre traced vertical strokes on their thin part only.

    A stem is traced together with whatever touches it: the notehead at one
    end (left of an up-stem's foot, right of a down-stem's head), a flag or
    beam at the other. Both pull the traced centre the same way, so stems on a
    flat page read ~1-2 deg off vertical (median -1 deg on the dev flats) --
    and the residual field, which trusts them, shears the page by that much.
    Rows where the ink run under the trace is wider than the page's stroke
    width (median run) + ``tol`` px are junctions; the stroke is re-centred
    from the remaining rows of the binary image. Strokes left shorter than
    ``min_span`` px are dropped."""
    R = [_stroke_runs(ink, p) for p in segs if len(p) >= 2]
    w = np.concatenate([r[:, 2] - r[:, 1] + 1 for r in R]) if R else np.zeros(0)
    w = w[np.isfinite(w)]
    if not len(w):
        return []
    w_max = float(np.median(w)) + tol
    out = []
    for r in R:
        ok = (r[:, 2] - r[:, 1] + 1) <= w_max      # NaN rows fail too
        if ok.sum() < 3:
            continue
        y, c = r[ok, 0], 0.5 * (r[ok, 1] + r[ok, 2])
        if y[-1] - y[0] < min_span:
            continue
        keep = np.unique(np.concatenate([np.arange(0, len(y), step), [len(y) - 1]]))
        out.append(np.stack([c[keep], y[keep]], 1))
    return out


def scale_structures(S: Structures, s: float) -> Structures:
    """Structures detected on one canvas, expressed on a canvas scaled by ``s``."""
    import copy
    T = copy.deepcopy(S)
    T.shape = (int(round(S.shape[0] * s)), int(round(S.shape[1] * s)))
    T.thickness, T.spacing_est = S.thickness * s, S.spacing_est * s
    for f in T.fragments:
        f.pts = f.pts * s
    for v in T.vsegs:
        v.pts = v.pts * s
    for st in T.staves:
        st.tuples = [t * s for t in st.tuples]
        st.spacing *= s
        st.x_range = (st.x_range[0] * s, st.x_range[1] * s)
        for name in ("top_poly", "bot_poly"):
            c = getattr(st, name)
            if c is not None:   # y' = s * f(x' / s): coefficient of x^k times s^(1-k)
                n = len(c) - 1
                setattr(st, name, np.array([ci * s ** (1 - (n - j)) for j, ci in enumerate(c)]))
    for r in T.rows:
        r.height *= s
        for b in r.blobs:
            b.x0, b.y0, b.x1, b.y1 = b.x0 * s, b.y0 * s, b.x1 * s, b.y1 * s
    T.regions = [(t, np.asarray(p, float) * s) for t, p in S.regions]
    if S.heads is not None:
        T.heads = S.heads.copy()
        T.heads[:, :3] *= s
    return T


def _learned_staves(seg, ink, thick, W, robust):
    """Staves from the network's staff-line mask (``learned.masks``).

    Returns (fragments, staves) or None when the mask holds no staff, so that
    jianpu / text pages keep the classical path unchanged. Horizontal lines
    outside the staff bands (jianpu 减时线, text underlines) still come from
    ``ink``; inside the bands they would be ledger lines / beams."""
    m = seg.get("staff")
    if m is None or m.sum() < 5e-4 * m.size:
        return None
    s_thick, s_sp = run_metrics(m)
    if not s_sp:
        return None
    sfr = [Fragment(p) for p in find_fragments(m, s_thick, s_sp)[0]]
    staves = group_staves(sfr, W, s_sp, bridge_gaps=robust)
    if not staves:
        return None
    sp = float(np.median([s.spacing for s in staves]))
    band = staff_band_mask(ink.shape, staves)
    # mask pieces outside any staff are mostly false positives (paper texture,
    # print-through): keep their slots (indices) but drop their evidence
    sfr = [f if f.staff >= 0 else Fragment(f.pts[:1]) for f in sfr]
    for p in find_fragments(ink, thick, sp)[0]:
        q = np.round(p[len(p) // 2]).astype(int)
        if not band[np.clip(q[1], 0, band.shape[0] - 1), np.clip(q[0], 0, band.shape[1] - 1)]:
            sfr.append(Fragment(p))
    return sfr, staves


def _staff_mid(st, xs):
    return 0.5 * (np.polyval(st.top_poly, xs) + np.polyval(st.bot_poly, xs))


def _same_staff(a, b):
    lo, hi = max(a.x_range[0], b.x_range[0]), min(a.x_range[1], b.x_range[1])
    if hi <= lo:
        return False
    xs = np.linspace(lo, hi, 9)
    return float(np.median(np.abs(_staff_mid(a, xs) - _staff_mid(b, xs)))) < 1.5 * min(a.spacing, b.spacing)


def _fuse_staves(cf, cs, lf, ls, prefer=1.3):
    """Per-staff fusion of classical (``cf``, ``cs``) and learned (``lf``,
    ``ls``) staves (fragments, staves); returns (fragments, staves, the kept
    learned staves).

    The network misses whole staves on low-resolution pages and loses pieces
    of a staff at strong curl; morphology misses grey staves under shading and
    breaks lines at dense notation. Learned staves are the base; a classical
    staff no learned staff overlaps is added (its line pieces had been free
    fragments of the learned path), and one that covers ``prefer`` times the
    line length of the learned staves it overlaps replaces them. Fragment
    indices are kept stable (removed pieces become 1-point placeholders)."""
    cov = lambda fr, st: sum(float(np.ptp(fr[i].pts[:, 0])) for i in st.frag_ids)
    frags = [Fragment(f.pts, f.staff, f.line) for f in lf]
    keep = [True] * len(ls)
    added = []
    for c in cs:
        m = [k for k, l_ in enumerate(ls) if keep[k] and _same_staff(c, l_)]
        if m and cov(cf, c) <= prefer * sum(cov(lf, ls[k]) for k in m):
            continue
        for k in m:
            keep[k] = False
            for fi in ls[k].frag_ids:
                frags[fi] = Fragment(frags[fi].pts[:1])
        # free pieces inside this staff's band are its own lines seen as free lines
        xs0, xs1 = c.x_range
        for fi, f in enumerate(frags):
            if f.staff >= 0 or len(f.pts) < 2:
                continue
            q = f.pts[len(f.pts) // 2]
            if xs0 - c.spacing <= q[0] <= xs1 + c.spacing:
                top = np.polyval(c.top_poly, q[0]) - 1.5 * c.spacing
                bot = np.polyval(c.bot_poly, q[0]) + 1.5 * c.spacing
                if top <= q[1] <= bot:
                    frags[fi] = Fragment(f.pts[:1])
        ids = []
        for fi in c.frag_ids:
            g = cf[fi]
            frags.append(Fragment(g.pts, -1, g.line))
            ids.append(len(frags) - 1)
        added.append(Staff(frag_ids=ids, tuples=c.tuples, spacing=c.spacing, top_poly=c.top_poly,
                           bot_poly=c.bot_poly, x_range=c.x_range))
    kept = [Staff(frag_ids=list(l_.frag_ids), tuples=l_.tuples, spacing=l_.spacing, top_poly=l_.top_poly,
                  bot_poly=l_.bot_poly, x_range=l_.x_range) for k, l_ in enumerate(ls) if keep[k]]
    staves = sorted(kept + added, key=lambda s_: np.polyval(s_.top_poly, np.mean(s_.x_range)))
    for f in frags:
        f.staff = -1
    for k, st in enumerate(staves):
        for fi in st.frag_ids:
            frags[fi].staff = k
    return frags, staves, kept


def spacing_classes(spacings, gap=1.3):
    """Label staves by printed size: sorted spacings split where one is more
    than ``gap`` times the next smaller one. A solo part above a piano score
    (or an ossia / cue staff) is printed at 0.55-0.75 of the main size, while
    perspective changes one size by at most ~1.2 across a photo. Priors of the
    form "all staves have the same spacing" (B, C) hold only within a class
    (hand-held violin + piano photos, 2026-11-19: one global spacing stretched
    the violin staves to the piano size and bent the whole page)."""
    sp = np.asarray(spacings, float)
    labels = np.zeros(len(sp), int)
    if len(sp) < 2:
        return labels
    order = np.argsort(sp)
    k = 0
    for prev, cur in zip(order[:-1], order[1:]):
        if sp[cur] > gap * sp[prev]:
            k += 1
        labels[cur] = k
    return labels


def detect_structures(norm: np.ndarray, ink: np.ndarray, *, robust=True, local_lines=False,
                      merge_rows=False, stem_refine=False, seg=None, valid=None, fuse="page",
                      jp_rows="chain", glyph="template") -> Structures:
    """Detect exported structures; ``robust=False`` preserves optimizer evidence.

    Recovered faint/broken lines are useful for page type and annotations, but
    do not yet have calibrated confidence for driving B/C's deformation fit.
    ``local_lines=True`` always also tries locally thresholded line evidence
    and keeps whichever finds more staff lines (thin grey lines next to dark
    noteheads are lost by global Otsu even when some staves survive).
    ``merge_rows=True`` joins jianpu / text row pieces split by wide gaps
    (``_merge_chains``) before rows are classified.
    ``stem_refine=True`` re-centres vertical strokes on their thin part
    (``refine_strokes``) on pages with staves: traced with noteheads / flags
    they lean ~1-2 deg. (Jianpu-only pages: no consistent gain on dev.)
    ``seg``: optional learned masks (``learned.masks``) on this image; staff
    lines then come from the staff mask and strokes inside staff bands from
    the stem mask (see ``_learned_staves``).
    ``jp_rows``: chain (all glyphs chained, rows classified afterwards) |
    digit (jianpu note rows chained from digit blobs first, ``find_text_rows``).
    ``glyph``: digit classifier for ``jp_rows="digit"``: template | cnn.
    ``valid``: optional 0/1 sheet mask; ink and learned masks off the sheet
    are ignored (``ink`` is expected to be masked already).
    """
    H, W = ink.shape
    if valid is not None:
        ink = ink & valid
        if seg is not None:
            seg = {k: v & valid for k, v in seg.items()}
    thick, sp = run_metrics(ink)
    frags_pts, hmask = find_fragments(ink, thick, sp if sp else 12, horizontal=True)
    frags = [Fragment(p) for p in frags_pts]
    staves = group_staves(frags, W, sp if sp else None, bridge_gaps=robust)
    n_lines = lambda st: sum(len(s.frag_ids) for s in st)
    if local_lines:
        block = max(15, min(H, W) // 50) | 1
        local_ink = cv2.adaptiveThreshold(norm, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, block, 5)
        if valid is not None:
            local_ink &= valid
        local_thick, local_sp = run_metrics(local_ink)
        local_frags = [Fragment(p) for p in find_fragments(local_ink, local_thick, local_sp or 12)[0]]
        local_staves = group_staves(local_frags, W, local_sp or None, bridge_gaps=robust)
        if n_lines(local_staves) > 1.1 * n_lines(staves):
            frags, staves, thick = local_frags, local_staves, local_thick
    elif robust and not staves:
        # Global Otsu is dominated by dark noteheads. On blurred/downsampled
        # scores the thin grey staff lines can be brighter than its threshold,
        # so no grouping algorithm can recover them from `ink`. Use local
        # contrast only for horizontal evidence; leave digit/text masks intact.
        block = max(15, min(H, W) // 50) | 1
        local_ink = cv2.adaptiveThreshold(norm, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, block, 5)
        if valid is not None:
            local_ink &= valid
        local_thick, local_sp = run_metrics(local_ink)
        local_pts, _ = find_fragments(local_ink, local_thick, local_sp or 12)
        local_frags = [Fragment(p) for p in local_pts]
        local_staves = group_staves(local_frags, W, local_sp or None)
        if local_staves:
            frags, staves = local_frags, local_staves
            thick = local_thick
    # learned staff mask: kept when it covers at least as much staff-line
    # length as the classical evidence (dev 2026-11-18: the mask alone lost whole staves
    # on low-resolution pages, recall 0.77 vs 0.88; positions equally accurate)
    # per-staff fusion: the network's clean mask is safe to bridge across
    # symbols / barlines (otherwise it splits a row into pieces at dense notation)
    learned = _learned_staves(seg, ink, thick, W, robust or fuse == "staff") if seg is not None else None
    # compare covered staff-line length, not fragment counts (the mask breaks
    # lines into more pieces at noteheads)
    cover = lambda fr, st: sum(float(np.ptp(fr[i].pts[:, 0])) for s_ in st for i in s_.frag_ids)
    stem_staves = None
    if fuse == "staff" and learned is not None:
        frags, staves, stem_staves = _fuse_staves(frags, staves, *learned)
        use_learned = bool(stem_staves)
    else:
        use_learned = learned is not None and cover(*learned) >= cover(frags, staves)
        if use_learned:
            frags, staves = learned
    if staves:
        sp = float(np.median([s.spacing for s in staves]))
    band = staff_band_mask(ink.shape, staves)
    rows, h_mode = find_text_rows(ink, int(max(25, 3 * (sp or 12))), band,
                                 reject_boxes=robust, merge_rows=merge_rows, digit_first=jp_rows == "digit", classifier=glyph)
    jp_rows = [r for r in rows if r.kind == "jianpu"]
    cands = []
    if staves:
        cands.append(2.2 * sp)
    if jp_rows:
        cands.append(1.1 * float(np.median([r.height for r in jp_rows])))
    elif h_mode:
        cands.append(1.1 * h_mode)
    Lv = max(16, int(min(cands))) if cands else 25
    vink = ink
    # stems only where the network also sees the staves (or the staff channel
    # is switched off for an ablation and the stem mask is not empty)
    if seg is not None and "stem" in seg and staves and (
            use_learned or ("staff" not in seg and seg["stem"].any())):
        # inside the staff bands only the network's stems / barlines (no
        # noteheads, flags, beams, slurs); jianpu / text strokes stay classical.
        # Its classes are exclusive, so stems are cut at every staff line:
        # add the staff pixels back (a horizontal line adds only its thickness
        # to a vertical run, so it bridges stems without creating verticals).
        bridge = seg.get("staff", seg.get("_staff_bridge"))
        stems = seg["stem"] | bridge if bridge is not None else seg["stem"]
        sband = band if stem_staves is None else staff_band_mask(ink.shape, stem_staves)
        vink = np.where(sband > 0, stems, ink).astype(np.uint8)
    vpts, _ = find_fragments(vink, thick, sp, horizontal=False, length=Lv)
    if stem_refine and staves:
        # the bias comes from noteheads / flags, so only pages with staves.
        # Refining only the strokes inside the staff bands was worse on dev
        # (paired AD +0.03 on staff and on mixed pages, 2026-11-18).
        vpts = refine_strokes(vpts, vink, min_span=0.55 * Lv)
    vsegs = []
    for p in vpts:
        if len(p) < 2:
            continue
        # straightness check
        A = np.stack([p[:, 1], np.ones(len(p))], 1)
        coef, *_ = np.linalg.lstsq(A, p[:, 0], rcond=None)
        if np.abs(A @ coef - p[:, 0]).max() < 2.0:
            vsegs.append(VSeg(p))
    S = Structures((H, W), thick, sp, frags, staves, rows, vsegs)
    S.heads = find_noteheads(ink, staves, sp)
    _classify_page(S)
    return S


def _classify_page(S: Structures):
    n_st = len(S.staves)
    n_jp = sum(r.kind == "jianpu" for r in S.rows)
    if n_st >= 1 and n_jp >= 1:
        S.page_type = "mixed"
    elif n_st >= 1:
        S.page_type = "staff"
    elif n_jp >= 1:
        S.page_type = "jianpu"
    else:
        S.page_type = "unknown"
    regs = []
    for st in S.staves:
        x0, x1 = st.x_range
        # Include interior extrema of curved staves, not just their endpoints.
        xs = np.linspace(x0, x1, 40)
        ys = np.concatenate([np.polyval(st.top_poly, xs), np.polyval(st.bot_poly, xs)])
        y0, y1 = ys.min() - 2 * st.spacing, ys.max() + 2 * st.spacing
        regs.append(("staff", np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)))
    for r in S.rows:
        x0 = min(b.x0 for b in r.blobs)
        x1 = max(b.x1 for b in r.blobs)
        y0 = min(b.y0 for b in r.blobs) - 0.4 * r.height
        y1 = max(b.y1 for b in r.blobs) + 0.4 * r.height
        regs.append(("jianpu" if r.kind == "jianpu" else "text", np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])))
    S.regions = regs


def draw_structures(img: np.ndarray, S: Structures) -> np.ndarray:
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for f in S.fragments:
        col = (0, 160, 0) if f.staff < 0 else (255, 80, 0)
        cv2.polylines(vis, [f.pts.astype(np.int32)], False, col, 2)
    for st in S.staves:
        for t in st.tuples[::3]:
            for p in t:
                cv2.circle(vis, tuple(int(v) for v in p), 2, (0, 0, 255), -1)
    for v in S.vsegs:
        cv2.polylines(vis, [v.pts.astype(np.int32)], False, (200, 0, 200), 2)
    for r in S.rows:
        col = (0, 0, 230) if r.kind == "jianpu" else (0, 140, 255)
        for k, b in enumerate(r.blobs):
            cv2.rectangle(vis, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), col, 1)
        bl = np.array([[b.cx, b.y1] for k, b in enumerate(r.blobs) if k in r.full])
        if len(bl) > 1:
            cv2.polylines(vis, [bl.astype(np.int32)], False, col, 2)
    cv2.putText(vis, f"type={S.page_type}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return vis
