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


def group_staves(frags, W, spacing_hint):
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


def classify_digit(crop):
    h, w = crop.shape
    asp = w / max(h, 1)
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


def find_text_rows(ink, line_len, band):
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
    hh = B[:, 3] - B[:, 1]
    cx = (B[:, 0] + B[:, 2]) / 2
    # pairwise right-neighbour search
    gap = B[None, :, 0] - B[:, None, 2]                         # x0_j - x1_i
    ov = np.minimum(B[:, None, 3], B[None, :, 3]) - np.maximum(B[:, None, 1], B[None, :, 1])
    mh = np.minimum(hh[:, None], hh[None, :])
    Mh = np.maximum(hh[:, None], hh[None, :])
    ok = (cx[None, :] > cx[:, None]) & (gap > -0.25 * mh) & (gap < 5.0 * Mh) & (ov > 0.5 * mh) & (Mh < 2.0 * mh)
    dist = np.where(ok, cx[None, :] - cx[:, None], np.inf)
    right = np.where(np.isfinite(dist.min(1)), dist.argmin(1), -1)
    left = np.where(np.isfinite(dist.min(0)), dist.argmin(0), -1)
    nxt = {i: int(right[i]) for i in range(len(blobs)) if right[i] >= 0 and left[right[i]] == i}
    has_prev = set(nxt.values())
    rows = []
    for i in range(len(blobs)):
        if i in has_prev:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        if len(chain) < 3:
            continue
        rb = [blobs[k] for k in chain]
        hmed = float(np.median([b.h for b in rb]))
        n_dig = 0
        for b in rb:
            crop = (lab[int(b.y0):int(b.y1), int(b.x0):int(b.x1)] == b._lab)
            if 0.75 * hmed <= b.h <= 1.3 * hmed:
                b.digit, _ = classify_digit(crop)
                n_dig += b.digit is not None
        kind = "jianpu" if (n_dig >= 4 and n_dig >= 0.55 * len(rb)) else "text"
        full = [k for k, b in enumerate(rb) if 0.8 * hmed <= b.h <= 1.25 * hmed]
        rows.append(TextRow(rb, kind, hmed, full))
    rows.sort(key=lambda r: np.median([b.cy for b in r.blobs]))
    return rows, h_mode


def detect_structures(norm: np.ndarray, ink: np.ndarray) -> Structures:
    H, W = ink.shape
    thick, sp = run_metrics(ink)
    frags_pts, hmask = find_fragments(ink, thick, sp if sp else 12, horizontal=True)
    frags = [Fragment(p) for p in frags_pts]
    staves = group_staves(frags, W, sp if sp else None)
    if staves:
        sp = float(np.median([s.spacing for s in staves]))
    band = staff_band_mask(ink.shape, staves)
    rows, h_mode = find_text_rows(ink, int(max(25, 3 * (sp or 12))), band)
    jp_rows = [r for r in rows if r.kind == "jianpu"]
    cands = []
    if staves:
        cands.append(2.2 * sp)
    if jp_rows:
        cands.append(1.1 * float(np.median([r.height for r in jp_rows])))
    elif h_mode:
        cands.append(1.1 * h_mode)
    Lv = max(16, int(min(cands))) if cands else 25
    vpts, _ = find_fragments(ink, thick, sp, horizontal=False, length=Lv)
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
        ys = np.concatenate([np.polyval(st.top_poly, [x0, x1]), np.polyval(st.bot_poly, [x0, x1])])
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
