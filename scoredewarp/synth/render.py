"""Synthetic flat score pages (staff / jianpu / mixed) with dense structural annotations.

Every drawn element is recorded as an annotation dict:
    {"kind": str, "pts": [[x, y], ...], ...meta}
Coordinates are in flat-page pixels. Kinds:
    staff_line   pts sampled along one staff line           meta: staff, line, spacing
    ledger       [left, right]
    barline      [top, bottom]                              meta: region ('staff'|'jianpu')
    stem         [top, bottom]
    notehead     [center]                                   meta: staff, note
    jp_char      [tl, tr, br, bl, top_c, bot_c, center]     meta: row, ch, idx, height
    jp_dash      [left, right]  (增时线)                     meta: row
    jp_dot       [center]       (高/低音点)                   meta: row, char_idx, pos
    jp_underline [left, right]  (减时线)                     meta: row, level
    lyric_char   [tl, tr, br, bl, top_c, bot_c, center]     meta: row, note_ref
    text_char    same as lyric_char (titles / headers)
    region       [tl, tr, br, bl]                           meta: type
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .fonts import cjk_font_paths, digit_font_paths, ink_bbox, load_font, symbol_font_paths

LYRICS = list("春眠不觉晓处闻啼鸟夜来风雨声花落知多少床前明月光疑是地上霜举头望低思故乡白日依山尽黄河入海流欲穷千里目更上一层楼")
TITLES = ["小星星", "茉莉花", "送别", "月亮代表我的心", "让我们荡起双桨", "雪绒花", "友谊地久天长", "康定情歌"]

PAGE_W, PAGE_H = 1240, 1754  # A4 @ 150 dpi


class Canvas:
    def __init__(self, w=PAGE_W, h=PAGE_H):
        self.img = Image.new("L", (w, h), 255)
        self.draw = ImageDraw.Draw(self.img)
        self.ann: list[dict] = []
        self.w, self.h = w, h

    def add(self, kind, pts, **meta):
        d = {"kind": kind, "pts": np.asarray(pts, float).reshape(-1, 2).tolist()}
        d.update(meta)
        self.ann.append(d)
        return len(self.ann) - 1

    # --- primitives --------------------------------------------------------
    def hline(self, x0, x1, y, w):
        # Filled rectangle centred on y so annotation y is the true centre line.
        self.draw.rectangle([x0, y - w / 2, x1, y + w / 2 - 1e-3], fill=0)

    def vline(self, x, y0, y1, w):
        self.draw.rectangle([x - w / 2, y0, x + w / 2 - 1e-3, y1], fill=0)

    def text_glyph(self, font_path, size, ch, cx, baseline):
        """Draw `ch` horizontally centred at cx on `baseline`; return ink box pts."""
        font = load_font(font_path, size)
        self.draw.text((cx, baseline), ch, font=font, fill=0, anchor="ms")
        x0, y0, x1, y1 = ink_bbox(font_path, size, ch, "ms")
        x0, x1, y0, y1 = cx + x0, cx + x1, baseline + y0, baseline + y1
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [xm, y0], [xm, y1], [xm, ym]]


def _font_size_for_height(path, target_h, probe="5"):
    ref = 100
    x0, y0, x1, y1 = ink_bbox(path, ref, probe, "ms")
    return max(6, int(round(ref * target_h / max(1, (y1 - y0)))))


# ---------------------------------------------------------------------------
# Staff notation block
# ---------------------------------------------------------------------------
def staff_block(cv: Canvas, rng, x0, x1, y_top, s, staff_id, lyrics=True, cjk=None, digit=None):
    thick = 2 if s >= 14 else int(rng.choice([1, 2]))
    y_bot = y_top + 4 * s
    xs = np.arange(x0, x1 + 1e-6, 10.0)
    for i in range(5):
        y = y_top + i * s
        cv.hline(x0, x1, y, thick)
        cv.add("staff_line", np.stack([xs, np.full_like(xs, y)], 1), staff=staff_id, line=i, spacing=s)

    # system start barline
    cv.vline(x0, y_top, y_bot, thick + 1)
    cv.add("barline", [[x0, y_top], [x0, y_bot]], region="staff")

    # clef
    sym = symbol_font_paths()
    xc = x0 + 0.6 * s
    if sym:
        size = int(7.5 * s)
        font = load_font(sym[0], size)
        bx0, by0, bx1, by1 = ink_bbox(sym[0], size, "\U0001D11E", "ls")
        # Align glyph so it spans ~[y_top-1.5s, y_bot+1.5s]
        target_top = y_top - 1.6 * s
        cv.draw.text((xc - bx0, target_top - by0), "\U0001D11E", font=font, fill=0, anchor="ls")
        xc += (bx1 - bx0) + 0.6 * s
    else:
        xc += 3 * s
    # time signature
    if digit:
        size = _font_size_for_height(digit, 1.9 * s)
        for k, yb in enumerate([y_top + 2 * s, y_bot]):
            cv.text_glyph(digit, size, "4", xc + 0.7 * s, yb - 0.05 * s)
        xc += 2.0 * s

    start = xc + 0.8 * s
    n_meas = int(rng.integers(3, 6))
    bars = np.linspace(start, x1, n_meas + 1)
    note_xs = []
    for m in range(n_meas):
        a, b = bars[m], bars[m + 1]
        cv.vline(b, y_top, y_bot, thick + 1)
        cv.add("barline", [[b, y_top], [b, y_bot]], region="staff")
        n = int(rng.choice([2, 3, 4, 4, 6, 8]))
        pad = 1.4 * s
        xsn = np.linspace(a + pad, b - pad, n) if n > 1 else np.array([(a + b) / 2])
        pitches = np.clip(np.cumsum(rng.integers(-3, 4, n)) + rng.integers(0, 9), -4, 12)
        beamed = n >= 6
        k = 0
        while k < n:
            group = [k, k + 1] if (beamed and k + 1 < n) else [k]
            ps = pitches[group]
            up = ps.mean() < 4
            ends = []
            for j in group:
                x, p = xsn[j], pitches[j]
                y = y_bot - p * s / 2
                _notehead(cv, x, y, s, hollow=(n <= 2 and rng.random() < 0.6), thick=thick)
                cv.add("notehead", [[x, y]], staff=staff_id)
                note_xs.append(x)
                # ledger lines
                for q in range(-2, p - 1, -2):
                    yl = y_bot - q * s / 2
                    cv.hline(x - 0.95 * s, x + 0.95 * s, yl, thick)
                    cv.add("ledger", [[x - 0.95 * s, yl], [x + 0.95 * s, yl]])
                for q in range(10, p + 1, 2):
                    yl = y_bot - q * s / 2
                    cv.hline(x - 0.95 * s, x + 0.95 * s, yl, thick)
                    cv.add("ledger", [[x - 0.95 * s, yl], [x + 0.95 * s, yl]])
                sx = x + 0.6 * s if up else x - 0.6 * s
                ends.append((sx, y, y - 3.5 * s if up else y + 3.5 * s))
                if rng.random() < 0.07:
                    acc = cjk or digit
                    if digit:
                        cv.text_glyph(digit, int(2.2 * s), "#", x - 1.6 * s, y + 0.7 * s)
            if len(group) == 2:
                (sx0, _, e0), (sx1, _, e1) = ends
                e0 = e1 = (min(e0, e1) if up else max(e0, e1))
                slope = float(np.clip((ps[1] - ps[0]) * 0.25 * s / 2, -0.6 * s, 0.6 * s))
                e0, e1 = e0 + slope / 2 * (-1), e1 + slope / 2
                ends = [(sx0, ends[0][1], e0), (sx1, ends[1][1], e1)]
                bt = 0.5 * s * (1 if up else -1)
                cv.draw.polygon([(sx0, e0), (sx1, e1), (sx1, e1 + bt), (sx0, e0 + bt)], fill=0)
            for sx, y, e in ends:
                cv.vline(sx, min(y, e), max(y, e), max(1, thick))
                cv.add("stem", [[sx, min(y, e)], [sx, max(y, e)]])
            k += len(group)

    bottom = y_bot + 2.5 * s
    if lyrics and cjk:
        lh = 1.7 * s
        size = _font_size_for_height(cjk, lh, "口")
        base = y_bot + 4.2 * s + lh
        last = -1e9
        for x in note_xs:
            if rng.random() < 0.9 and x - last > 1.15 * lh:
                last = x
                ch = LYRICS[int(rng.integers(len(LYRICS)))]
                pts = cv.text_glyph(cjk, size, ch, x, base)
                cv.add("lyric_char", pts, row=f"staff{staff_id}_lyr", note_ref=float(x))
        bottom = base + 0.6 * s
    cv.add("region", [[x0, y_top - 2 * s], [x1, y_top - 2 * s], [x1, bottom], [x0, bottom]], type="staff")
    return bottom


def _notehead(cv, x, y, s, hollow, thick):
    t = np.linspace(0, 2 * np.pi, 40)
    a, b, ang = 0.66 * s, 0.47 * s, np.deg2rad(-22)
    px = x + a * np.cos(t) * np.cos(ang) - b * np.sin(t) * np.sin(ang)
    py = y + a * np.cos(t) * np.sin(ang) + b * np.sin(t) * np.cos(ang)
    poly = list(zip(px, py))
    if hollow:
        cv.draw.polygon(poly, outline=0, fill=255)
        cv.draw.line(poly + [poly[0]], fill=0, width=max(2, thick + 1))
    else:
        cv.draw.polygon(poly, fill=0)


# ---------------------------------------------------------------------------
# Jianpu (numbered notation) block
# ---------------------------------------------------------------------------
def jianpu_block(cv: Canvas, rng, x0, x1, base, h, row_id, digit, cjk, n_lyrics=1):
    size = _font_size_for_height(digit, h)
    lw = max(1.5, 0.075 * h)
    beats = int(rng.choice([2, 3, 4, 4]))
    n_meas = int(rng.integers(3, 6)) if beats <= 3 else int(rng.integers(3, 5))
    total = beats * n_meas
    slot = (x1 - x0 - 0.4 * h) / total
    top = base - h
    bar_top, bar_bot = base - 1.45 * h, base + 0.45 * h
    chars = []  # (idx, x, is_note)
    cx = x0 + 0.2 * h
    prev_quarter = False
    for m in range(n_meas):
        for b in range(beats):
            choice = rng.random()
            if prev_quarter and choice < 0.15:
                kind = "dash"
            elif choice < 0.55:
                kind = "q"
            elif choice < 0.85 or slot < 3.6 * h:
                kind = "e"
            else:
                kind = "s"
            n = {"dash": 0, "q": 1, "e": 2, "s": 4}[kind]
            if kind == "dash":
                xm = cx + slot / 2
                ym = base - 0.5 * h
                cv.hline(xm - 0.35 * h, xm + 0.35 * h, ym, lw * 1.1)
                cv.add("jp_dash", [[xm - 0.35 * h, ym], [xm + 0.35 * h, ym]], row=row_id)
            else:
                xs = cx + slot * (np.arange(n) + 0.5) / n
                group = []
                for x in xs:
                    ch = str(int(rng.integers(0, 8))) if rng.random() < 0.12 else str(int(rng.integers(1, 8)))
                    pts = cv.text_glyph(digit, size, ch, x, base)
                    idx = cv.add("jp_char", pts, row=row_id, ch=ch, height=pts[3][1] - pts[0][1])
                    chars.append((idx, x, ch != "0"))
                    group.append((idx, x, ch))
                n_lines = {"q": 0, "e": 1, "s": 2}[kind]
                for lv in range(n_lines):
                    yl = base + 0.22 * h + lv * 0.2 * h
                    xa, xb = xs[0] - 0.33 * h, xs[-1] + 0.33 * h
                    cv.hline(xa, xb, yl, lw)
                    cv.add("jp_underline", [[xa, yl], [xb, yl]], row=row_id, level=lv)
                for idx, x, ch in group:
                    if ch == "0":
                        continue
                    r = rng.random()
                    rad = max(1.6, 0.085 * h)
                    if r < 0.13:
                        yd = top - 0.28 * h
                        pos = "above"
                    elif r < 0.26:
                        yd = base + 0.22 * h + n_lines * 0.2 * h + 0.12 * h
                        pos = "below"
                    else:
                        continue
                    cv.draw.ellipse([x - rad, yd - rad, x + rad, yd + rad], fill=0)
                    cv.add("jp_dot", [[x, yd]], row=row_id, char_idx=idx, pos=pos)
            prev_quarter = kind == "q"
            cx += slot
        bx = cx if m < n_meas - 1 else x1
        cv.vline(bx, bar_top, bar_bot, max(1.5, 0.07 * h))
        cv.add("barline", [[bx, bar_top], [bx, bar_bot]], region="jianpu")
        if m == n_meas - 1:  # final double bar
            cv.vline(bx - 0.25 * h, bar_top, bar_bot, max(1.5, 0.07 * h))
            cv.add("barline", [[bx - 0.25 * h, bar_top], [bx - 0.25 * h, bar_bot]], region="jianpu")
    bottom = bar_bot
    lh = 0.85 * h
    lsize = _font_size_for_height(cjk, lh, "口")
    for r in range(n_lyrics):
        lbase = base + 0.95 * h + (r + 1) * 1.45 * lh
        last = -1e9
        for idx, x, is_note in chars:
            if is_note and rng.random() < 0.93 and x - last > 1.1 * lh:
                last = x
                ch = LYRICS[int(rng.integers(len(LYRICS)))]
                pts = cv.text_glyph(cjk, lsize, ch, x, lbase)
                cv.add("lyric_char", pts, row=f"{row_id}_lyr{r}", note_ref=idx)
        bottom = lbase + 0.25 * lh
    cv.add("region", [[x0, bar_top - 0.3 * h], [x1, bar_top - 0.3 * h], [x1, bottom], [x0, bottom]], type="jianpu")
    return bottom


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render_page(kind: str, rng: np.random.Generator):
    """kind in {'staff', 'jianpu', 'mixed'}. Returns (uint8 image, annotations, meta)."""
    cv = Canvas()
    digit = digit_font_paths()[int(rng.integers(len(digit_font_paths())))]
    cjk = cjk_font_paths()[int(rng.integers(len(cjk_font_paths())))]
    mx = int(rng.integers(80, 130))
    x0, x1 = mx, PAGE_W - mx

    title = TITLES[int(rng.integers(len(TITLES)))]
    tsize = _font_size_for_height(cjk, 40, "口")
    tw = 46 * len(title)
    for i, ch in enumerate(title):
        pts = cv.text_glyph(cjk, tsize, ch, PAGE_W / 2 - tw / 2 + 23 + 46 * i, 120)
        cv.add("text_char", pts, row="title")
    if kind in ("jianpu", "mixed"):
        hsize = _font_size_for_height(digit, 22)
        cx = x0 + 10
        for ch in "1=C  4/4":
            if ch != " ":
                pts = cv.text_glyph(digit, hsize, ch, cx, 175)
                cv.add("text_char", pts, row="header")
            cx += 15

    s = float(rng.uniform(12, 17))
    h = float(rng.uniform(22, 32))
    y = 230.0
    bottom_margin = 110
    seq = []
    if kind == "mixed":
        first = rng.random() < 0.5
        seq = ["staff" if (i % 2 == 0) == first else "jianpu" for i in range(20)]
    else:
        seq = [kind] * 20
    staff_id = row_id = 0
    for blk in seq:
        if blk == "staff":
            need = 4 * s + 2 * s + (6 * s)
            if y + 2 * s + need > PAGE_H - bottom_margin:
                break
            lyr = rng.random() < 0.7
            y_top = y + 2.2 * s
            y = staff_block(cv, rng, x0, x1, y_top, s, staff_id, lyrics=lyr, cjk=cjk, digit=digit)
            staff_id += 1
            y += float(rng.uniform(2.5, 4.0)) * s
        else:
            nl = int(rng.choice([1, 1, 2]))
            need = 1.6 * h + 0.95 * h + nl * 1.45 * 0.85 * h + 0.3 * h
            if y + need > PAGE_H - bottom_margin:
                break
            base = y + 1.6 * h
            y = jianpu_block(cv, rng, x0, x1, base, h, f"jp{row_id}", digit, cjk, n_lyrics=nl)
            row_id += 1
            y += float(rng.uniform(0.9, 1.6)) * h
    img = np.asarray(cv.img, dtype=np.uint8).copy()
    meta = {"kind": kind, "staff_spacing": s, "digit_height": h, "digit_font": digit, "cjk_font": cjk,
            "size": [PAGE_W, PAGE_H]}
    return img, cv.ann, meta
