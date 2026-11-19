"""Main correction pipeline: A (page + deskew) -> metric B (camera + GCS) -> residual.

  A  page quadrilateral homography + projection-profile deskew (unchanged)
  B  developable general-cylinder surface + pinhole camera fitted to staff
     lines, jianpu double lines, text rows, vertical strokes and justified
     system ends; flattened isometrically over the whole visible page
  R  optional residual on the B output: the linear field (legacy C) or the
     bounded staff-line mesh
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .field import margin_stats
from .gcs import GCSModel
from .page import (detect_page_quad, estimate_skew, page_edges, page_mask, quad_homography, refine_page_mask,
                   rotation_homography)
from .pipeline import Config, _detect, _finish, _warp_mask, _warp_seg
from .preprocess import binarize, content_blobs, normalize_illumination, to_gray
from .structures import scale_structures
from .transforms import BackwardMap, native_scale, photo_footprint


# Tuned on the Doc3D-retextured dev split (data/bench/doc3d_score/dev), see
# reports/metric_route_20261117.md. Vertical strokes need a high weight: they are
# the only evidence against shear, and horizontal lines outnumber them -- so
# they must be unbiased: stems traced with their noteheads / flags lean ~1 deg
# (MetricConfig.stem_refine). The
# residual field is smoothed per component: V is pinned by many reliable staff
# lines (soft), U only by short, noisier strokes (stiffer).
DEFAULT_WEIGHTS = {"vseg": 5.0}
# 2026-11-18 (dev): every staff / jianpu row is pulled to the page's spacing /
# digit height (spacing_gate 9: the old 20 % gate kept exactly the worst-fitted
# staves as they were), the staves' left / right ends (clefs, final barlines)
# share one U (ends_w). Per page type: jianpu barlines and upright character
# strokes are reliable verticals (vseg_w 8 -- on staff pages short accidental /
# beam fragments are not, and AD suffers); on mixed pages a softer U lets the
# short jianpu barlines straighten (a softer U on jianpu pages costs AD).
# keep_scale: the field must not change the page's global scale / aspect, which
# the bench cannot see (its alignment fits X and Y scale separately); anchor:
# nor slide content off the canvas.
# 2026-11-18 (dev, round 4): the staves' ends per staff ROW from the line
# fragments, indented / short staves dropped (ends_mode rows): the 5-line
# tuples stop short of the ends and split staves broke the chain, so the
# margins kept a slant / keystone (the "top wide, bottom narrow" CPMS pages).
DEFAULT_FIELD = {"tuple_u": 0.0, "vseg_w": 4.0, "lam_u": 3.0, "lam_v": 0.5, "spacing_gate": 9.0, "ends_w": 2.0,
                 "ends_mode": "rows", "keep_scale": True, "anchor": True, "guard": True,
                 # lengths along staves / jianpu rows from glyph widths (noteheads, same-class digits), not from the
                 # vertical spacing: dev 2026-11-19 ad_c median 0.875 -> 0.742, 72 better / 25 worse
                 "h_ruler": "glyph",
                 # with a true horizontal ruler the jianpu row lengths deserve weight next to the vertical strokes
                 # (jianpu + mixed dev 48: ad_c 37 better / 5 worse; 8.0 saturates); lyric / text rows keep 0.3
                 "row_u": 3.0, "text_u": 0.3,
                 # the text block's sides are straight lines on the flat page: staff row ends + jianpu final
                 # barlines (block_jp) share one U per side, also along the segments between rows (edge_w)
                 "block_jp": True, "edge_w": 2.0}
DEFAULT_FIELD_TYPE = {"jianpu": {"vseg_w": 8.0}, "mixed": {"lam_u": 1.5}}   # on top of DEFAULT_FIELD


@dataclass
class MetricConfig:
    coarse_width: int = 1600
    robust_detect: bool = False
    local_lines: bool = True
    stem_refine: bool = True         # vertical strokes re-centred on their thin part (no notehead / flag pull)
    gcs_ends: str = "all"            # all: every outer-line fragment end | rows: only each staff row's outer ends
    jp_rows: str = "chain"           # chain | digit: jianpu note rows chained from digit blobs first
    glyph: str = "template"          # template | cnn: digit classifier for jp_rows=digit (core/glyphnet.py)
    merge_rows: bool = False         # join jianpu / text row pieces split by wide gaps
    adaptive_scale: str = "staff"   # staff: upscale when detected staves are small | off
    full_page: bool = True
    edges: bool = True               # use the paper borders (needs full_page)
    edge_mode: str = "extent"        # extent: borders bound the output only | fit: also straight-line constraints
    edge_w: float = 1.0
    edge_tol: float = 0.006          # RANSAC tolerance, fraction of the content size
    edge_max_err: float = 4.0        # px; a side fitted worse than this is not used for the extent
    edge_widen: bool = True          # re-space the slope knots over the borders
    focal_scale: float = 1.0
    ink_extent: bool = True          # output holds all printed content on the sheet, not only the structures
    margin: float = 0.03             # min. content-to-border distance, fraction of the content width
    rectify_margins: bool = False    # final per-row stretch of the staves' ends onto two vertical lines (neutral on dev)
    rect_max_deg: float = 2.0        # ... only for a margin slant up to this
    rect_max_kst: float = 0.05       # ... and a top -> bottom width change up to this
    rect_min_rows: int = 4
    residual: str = "field"          # none | field | mesh
    output_res: str = "native"       # native: final output sampled at the photo's own resolution | canvas
    seg: str = "off"                 # off | homr: learned staff-line / stem masks replace morphology (core/learned.py)
    seg_parts: str | None = None     # e.g. staff / stem / staff+stem; None also honours SEG_PARTS env
    stem_bridge: bool = True         # preserve staff crossings in the stem-only ablation
    seg_fuse: str = "page"           # page: learned or classical staves per page (by line length) | staff: per staff
    seg_stages: str = "bcf"          # where the learned masks are used: b surface fit, c residual field, f final output
    page_seg: str = "off"            # off | u2netp: learned sheet mask, evidence only on the sheet (core/pagenet.py)
    page_rim: bool = True            # with a sheet mask: drop background slivers along its border (pipeline.drop_rim)
    page_erode: float = 0.01         # sheet mask eroded by this fraction of the photo width (paper rim, shadow edge)
    page_seg_extent: bool = False    # also use the learned sheet mask in place of the brightness mask for the extent
    page_quad: bool = False          # with page_seg: page quad from the learned mask when the brightness quad fails
    rot90: bool = False              # turn the page by 90 degrees when its staves run vertically (_turned_staves)
    main_page: bool = False          # crop the output to the main page: block + its print + margin (mainpage.py)
    page_select: bool = True         # with page_seg: drop the facing page of a two-page spread (pagenet.spread_cut)
    extent_fold: bool = False        # the output extent stops where the fitted surface folds over (GCSModel._unfolded)
    edge_prefit: bool = False        # with a learned sheet mask: its four sides are straight-line constraints of B
    edge_prefit_w: float = 1.0       # from the first fit on (soft_l1: curled stretches act as outliers)
    sheet_content: bool = True       # with a learned sheet mask: all print on the sheet is content of the output
    quad_alt: bool = False           # with a learned page quad: also fit without it, keep the more rectangular outline
    b_select: str = "cost"           # cost | outline | outline_margin: with a learned sheet mask, finish both ruling
                                     # starts, keep the most rectangular sheet outline (GCSModel.select_by_outline);
                                     # outline_margin also wants constant page margins to the staff block
    pool: str = "off"                # off | width: run the whole pipeline at each of ``pool_widths`` (the B fit
                                     # lands in a different local minimum on ~1 page in 4 when the coarse frame
                                     # moves by 1 %) and keep the run whose residual field C had the least work
                                     # to do: C absorbs what B got wrong, so its total displacement is a GT-free
                                     # error estimate of B (tune: mean AD_c -11 %, big gains 12 vs big losses 1)
    pool_widths: tuple = (1600, 1584, 1616)
    field_iters: int = 3
    fit_kw: dict = None
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    field_kw: dict = field(default_factory=lambda: dict(DEFAULT_FIELD))
    field_type: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_FIELD_TYPE.items()})


def _coarse_homography(img, quad, width):
    H0, W0 = img.shape[:2]
    if quad is not None:
        return quad_homography(quad, width)
    sc = width / W0
    return np.diag([sc, sc, 1.0]), (int(W0 * sc), int(H0 * sc))


def _turned_staves(I1, ratio=2.0):
    """True when staff lines run vertically: the staves found on the image
    turned by 90 degrees are much longer than those found as is. Only staves
    vote (jianpu digit rows are found in both directions); the image is left as
    is when neither direction has staves. 180 degrees is not decided."""
    def length(im):
        _, _, S = _detect(im, robust=False)
        return float(sum(s.x_range[1] - s.x_range[0] for s in S.staves))
    turned = length(np.ascontiguousarray(np.rot90(I1, -1)))
    return turned > 0 and turned > ratio * length(I1)


def coarse(img, cfg, sal=None, stages=None):
    """``sal``: learned saliency map (``pagenet.saliency``); with
    ``cfg.page_quad`` its page quadrilateral replaces a missing brightness quad."""
    quad = detect_page_quad(img)
    if quad is None and sal is not None and cfg.page_quad:
        from .pagenet import page_quad
        quad = page_quad(img, sal)
        if stages is not None:
            stages["quad_source"] = "pagenet" if quad is not None else None
    Hq, size = _coarse_homography(img, quad, cfg.coarse_width)
    I1 = cv2.warpPerspective(img, Hq, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    r90 = cfg.rot90 and _turned_staves(I1)
    if stages is not None and cfg.rot90:
        stages["rot90"] = bool(r90)
    I1r = np.ascontiguousarray(np.rot90(I1, -1)) if r90 else I1
    ang, _ = estimate_skew(binarize(normalize_illumination(to_gray(I1r))))
    R, size = rotation_homography(ang, size, r90)
    Hc = R @ Hq
    I1 = cv2.warpPerspective(img, Hc, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Hc, size, I1, quad, ang


def detect_at_scale(img, Hc, size, I1, cfg, stages, seg_src=None, page_src=None):
    """Detect structures; redo at a larger coarse scale for small staff lines.

    The detector works in integer pixels (run lengths >= 6 px). Dense scores
    photographed small (e.g. orchestral pages) fall below that, so when staves
    are found with spacing < 9 px the coarse image is re-rendered with spacing
    ~12 px. Only measured staff spacing triggers this: other periodic ink
    (digit strokes) would upscale jianpu pages for nothing, which hurts them.
    """
    _, _, S = _detect(I1, robust=cfg.robust_detect, local_lines=cfg.local_lines, merge_rows=cfg.merge_rows,
                      stem_refine=cfg.stem_refine, seg=_warp_seg(BackwardMap.from_homography(Hc, size), seg_src),
                      seg_parts=cfg.seg_parts, stem_bridge=cfg.stem_bridge,
                      page=_warp_mask(BackwardMap.from_homography(Hc, size), page_src), seg_fuse=cfg.seg_fuse,
                      rim=cfg.page_rim, jp_rows=cfg.jp_rows, glyph=cfg.glyph)
    if cfg.adaptive_scale != "off" and S.staves:
        sp = float(np.median([st.spacing for st in S.staves]))
        if sp < 9:
            f = min(12.0 / sp, 2.5)
            Hc = np.diag([f, f, 1.0]) @ Hc
            size = (int(round(size[0] * f)), int(round(size[1] * f)))
            I1 = cv2.warpPerspective(img, Hc, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            _, _, S = _detect(I1, robust=cfg.robust_detect, local_lines=cfg.local_lines, merge_rows=cfg.merge_rows,
                              stem_refine=cfg.stem_refine,
                              seg=_warp_seg(BackwardMap.from_homography(Hc, size), seg_src),
                              seg_parts=cfg.seg_parts, stem_bridge=cfg.stem_bridge,
                              page=_warp_mask(BackwardMap.from_homography(Hc, size), page_src), seg_fuse=cfg.seg_fuse,
                      rim=cfg.page_rim, jp_rows=cfg.jp_rows, glyph=cfg.glyph)
            stages["coarse_upscale"] = f
    return Hc, size, I1, S


def grow_to_content(model, img, extent, spacing, margin=0.0, look=0.15, scale=0.5, sheet=None):
    """Grow the flat output box over all printed content on the sheet.

    The surface is fitted to staff lines / rows / strokes, and the page box is
    limited by the paper mask; notes and hairpins below the last staff, a lone
    exercise number or a sheet running out of a shaded photo edge lie outside
    both. A half-resolution preview of the box widened by ``look`` is searched
    for print-like blobs mostly inside the box (``content_blobs``, as for the
    reading copy); the box then holds their ink plus ``margin`` (fraction of the
    content width) on every side. The pipeline adds its margin only after the
    residual (``_finish``), so it calls this with ``margin=0``.
    Returns (box, outline points of the content in the output pixels of the
    returned box, or None).
    ``sheet``: learned sheet mask on the photo; ink on it counts as content
    even outside the box (a page number in a curled corner that the fitted
    border line cuts off).
    """
    x0, x1, y0, y1 = extent
    gx, gy = look * (x1 - x0), look * (y1 - y0)
    big = (x0 - gx, x1 + gx, y0 - gy, y1 + gy)
    ppu = model.unit * scale
    bmp = model.backward_map(big, out_scale=scale)
    # shrink the photo first (area filter) so thin lines survive the remap
    f = min(1.0, 1.5 * bmp.out_size[0] / img.shape[1])
    src = cv2.resize(to_gray(img), None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else to_gray(img)
    prev = BackwardMap(bmp.grid * f, bmp.stride, bmp.out_size).apply(src, interp=cv2.INTER_LINEAR, value=255)
    region = np.zeros(prev.shape, np.uint8)
    region[int(gy * ppu):int((gy + y1 - y0) * ppu), int(gx * ppu):int((gx + x1 - x0) * ppu)] = 255
    # Beyond the visible sheet the surface is extrapolated and may run back
    # onto photo pixels the page box already shows (a second copy of the score
    # past a fold): outside the box, keep only points whose photo pixels are
    # not in the box's footprint.
    g = bmp.grid
    gh, gw = g.shape[:2]
    oy, ox = np.mgrid[0:gh, 0:gw] * bmp.stride
    inbox = region[np.clip(oy, 0, region.shape[0] - 1), np.clip(ox, 0, region.shape[1] - 1)] > 0
    gap = bmp.stride * img.shape[1] / max(bmp.out_size[0], 1)
    dup = photo_footprint(g[inbox], img.shape[1::-1], gap)(g)
    bad = (~inbox & dup).astype(np.uint8)
    bad = cv2.resize(bad, prev.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    prev[bad & (region == 0)] = 255
    if sheet is not None:
        sh = cv2.resize(sheet, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else sheet
        on = BackwardMap(bmp.grid * f, bmp.stride, bmp.out_size).apply(sh, interp=cv2.INTER_LINEAR, value=0) > 127
        region = np.where(on & ~bad, 255, region).astype(np.uint8)
    # off the photo the preview is white; the shading flattening turns the
    # dark background next to it into a false ink line -> ignore that rim
    inph = BackwardMap(bmp.grid, bmp.stride, bmp.out_size).full_maps()
    inph = (inph[0] >= 0) & (inph[0] <= img.shape[1] - 1) & (inph[1] >= 0) & (inph[1] <= img.shape[0] - 1)
    r = max(3, int(round(2 * spacing * scale))) | 1
    valid = cv2.erode(inph.astype(np.uint8), np.ones((r, r), np.uint8))
    # where the extrapolated surface spreads few photo pixels over many output
    # pixels (past a curled edge) the paper rim is smeared into ink-like streaks
    sx = np.linalg.norm(np.diff(g, axis=1), axis=-1)[:-1]
    sy = np.linalg.norm(np.diff(g, axis=0), axis=-1)[:, :-1]
    sc = np.nan_to_num(np.minimum(sx, sy), nan=0.0)
    ref = np.median(sc[inbox[:-1, :-1] & (sc > 0)]) if (inbox[:-1, :-1] & (sc > 0)).any() else 0.0
    if ref > 0:
        ok = cv2.resize((sc >= 0.4 * ref).astype(np.uint8), prev.shape[::-1], interpolation=cv2.INTER_NEAREST)
        valid = ((valid > 0) & (ok > 0)).astype(np.uint8)
    blobs, ink = content_blobs(prev, region, spacing * scale, min_thick=1.0, min_size=1.0, valid=valid)
    on = (blobs > 0) & (ink > 0)
    if on.sum() < 50:
        return extent, None
    ys, xs = np.nonzero(on)
    bx0, bx1 = big[0] + xs.min() / ppu, big[0] + (xs.max() + 1) / ppu
    by0, by1 = big[2] + ys.min() / ppu, big[2] + (ys.max() + 1) / ppu
    m = margin * (bx1 - bx0)
    box = (min(x0, bx0 - m), max(x1, bx1 + m), min(y0, by0 - m), max(y1, by1 + m))
    # outline of the content (ink closed over each blob) in output pixels of box
    k = max(5, int(round(1.2 * spacing * scale))) | 1
    solid = cv2.erode(blobs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    solid[on] = 255
    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.concatenate([c.reshape(-1, 2) for c in cnts]).astype(float)
    pts = (pts / ppu + [big[0] - box[0], big[2] - box[2]]) * model.unit
    return box, pts


def _surface(img, cfg, stages, T, Hc, size, I1, seg_b, page_src, sal):
    """Detection (with the learned sheet mask when configured) and the fitted
    GCS surface in the coarse frame ``Hc``: one surface hypothesis (dict)."""
    H0, W0 = img.shape[:2]
    score = None
    Hc0, size0, I10 = Hc, size, I1
    Hc, size, I1, S1 = detect_at_scale(img, Hc, size, I1, cfg, stages, seg_b, page_src)
    sheet = None
    if page_src is None and cfg.page_seg == "u2netp":
        from .pagenet import content_points, cut_mask, sheet_mask, spread_cut
        t = time.time()
        sheet, info = sheet_mask(img, content_points(S1, Hc), sal=sal)
        stages["page_seg"] = info
        if sheet is not None:
            k = max(3, int(round(cfg.page_erode * W0))) | 1
            page_src = cv2.erode(sheet, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            stages.pop("coarse_upscale", None)
            Hc, size, I1, S1 = detect_at_scale(img, Hc0, size0, I10, cfg, stages, seg_b, page_src)
            cut = spread_cut(S1) if cfg.page_select else None
            if cut is not None:
                keep = cut_mask(img.shape, cut, Hc, S1.shape)
                sheet, page_src = sheet & keep, page_src & keep
                info["spread_cut"] = {"x": round(float(cut[0]), 1), "deg": round(float(np.degrees(cut[1])), 1),
                                      "keep": "left" if cut[2] else "right"}
                stages.pop("coarse_upscale", None)
                Hc, size, I1, S1 = detect_at_scale(img, Hc0, size0, I10, cfg, stages, seg_b, page_src)
        T["page_seg"] = time.time() - t
    stages["structures_coarse"] = S1.summary()
    model = GCSModel((W0, H0), focal=cfg.focal_scale * max(W0, H0))
    n = model.build(S1, Hc, weights=cfg.weights, ends_mode=cfg.gcs_ends)
    stages["surface_observations"] = int(n)
    if n >= 20:
        fit_kw = dict(cfg.fit_kw or {})
        outline = None
        if cfg.b_select in ("outline", "outline_margin") and sheet is not None:
            from .pagenet import outline_points
            outline = outline_points(sheet)
            if outline is not None:
                fit_kw["finish_all"] = True
        if cfg.edge_prefit and sheet is not None:
            # the paper border of the learned mask (RDGR-style boundary terms):
            # added before the first fit, so that a wrong perspective / ruling
            # solution is not reached first and the border then judged by it
            cand0 = page_edges(sheet, Hc, img=img)
            if len(cand0) >= 2:
                model.p = model.p0.copy()
                if model.add_edges(cand0, Hc, w=cfg.edge_prefit_w, widen=cfg.edge_widen):
                    stages["edges_prefit"] = {k: len(v) for k, v in cand0.items()}
        model.fit(**fit_kw)
        if outline is not None:
            stages["b_select"] = model.select_by_outline(outline, margin=cfg.b_select == "outline_margin")
        if sheet is not None and (cfg.quad_alt or outline is not None):
            from .pagenet import outline_points
            q = outline if outline is not None else outline_points(sheet)
            sc = model.outline_score(q) if q is not None else float("nan")
            if cfg.b_select == "outline_margin" and q is not None and np.isfinite(sc):
                mg = model.margin_score(q)
                sc = sc + mg if np.isfinite(mg) else sc
            score = sc if np.isfinite(sc) else None
        elif sheet is not None:
            from .pagenet import outline_points
            q = outline_points(sheet)
            sc = model.outline_score(q) if q is not None else float("nan")
            score = sc if np.isfinite(sc) else None
        stages["outline_score"] = score
    return {"Hc": Hc, "size": size, "I1": I1, "S1": S1, "sheet": sheet, "page_src": page_src, "model": model,
            "n": n, "outline_score": score}


def dewarp_metric(img: np.ndarray, cfg: MetricConfig | None = None, verbose=False, page_src=None):
    """``page_src``: optional uint8 sheet mask on the photo (255 = sheet); the
    detector then only takes evidence on the sheet. ``cfg.pool="width"``: see
    ``MetricConfig.pool``."""
    cfg = cfg or MetricConfig()
    if cfg.pool == "off":
        return _dewarp_single(img, cfg, verbose, page_src)
    if cfg.pool != "width":
        raise ValueError(f"unknown pool mode {cfg.pool!r}")
    t0 = time.time()
    base = replace(cfg, pool="off", output_res="canvas", main_page=False)
    cands, results = [], []
    for w in cfg.pool_widths:
        r = _dewarp_single(img, replace(base, coarse_width=int(w)), False, page_src)
        fd = r.stages.get("field_disp")
        cands.append({"width": int(w), "disp": float(sum(fd)) if fd else None})
        results.append(r)
    disp = [np.inf if c["disp"] is None else c["disp"] for c in cands]
    k = int(np.argmin(disp))     # ties (no field anywhere) keep the first width
    res = results[k]
    res.stages["pool"] = {"candidates": cands, "selected": k}
    res.timings["pool_total"] = time.time() - t0
    if cfg.output_res == "native":
        paper = res.paper
        res = to_native(img, res)
        if paper is not None and paper.shape[:2] != res.image.shape[:2]:
            res.paper = cv2.resize(paper, res.bmap.out_size, interpolation=cv2.INTER_NEAREST)
    return _main_page(res, cfg)


def _dewarp_single(img: np.ndarray, cfg: MetricConfig, verbose=False, page_src=None):
    t0 = time.time()
    T, stages = {}, {"model": "gcs"}
    H0, W0 = img.shape[:2]
    sal = None
    if cfg.page_seg == "u2netp":
        from .pagenet import saliency
        t = time.time()
        sal = saliency(img)
        T["saliency"] = time.time() - t
    Hc, size, I1, quad, ang = coarse(img, cfg, sal, stages)
    bm = BackwardMap.from_homography(Hc, size)
    cur = I1
    stages["quad"] = None if quad is None else quad.tolist()
    stages["skew_deg"] = float(ang)
    T["coarse"] = time.time() - t0

    seg_src = None
    if cfg.seg == "homr":
        from .learned import segment
        t = time.time()
        seg_src = segment(img)
        T["segment"] = time.time() - t
    t = time.time()
    seg_b = seg_src if "b" in cfg.seg_stages else None
    hyp = _surface(img, cfg, stages, T, Hc, size, I1, seg_b, page_src, sal)
    if cfg.quad_alt and stages.get("quad_source") == "pagenet" and hyp["outline_score"] is not None:
        # the learned quad gives a better coarse frame on most pages that lack a
        # brightness quad, but on curled sheets a worse one: fit the surface in
        # both frames and keep the more rectangular sheet outline
        st2, T2 = {}, {}
        Hb, sb, Ib, _, _ = coarse(img, replace(cfg, page_quad=False), None, None)
        alt = _surface(img, cfg, st2, T2, Hb, sb, Ib, seg_b, page_src, sal)
        choice = {"pagenet_quad": hyp["outline_score"], "no_quad": alt["outline_score"]}
        if alt["outline_score"] is not None and alt["outline_score"] < hyp["outline_score"]:
            hyp = alt
            for k in [k for k in stages if k not in ("model", "quad", "skew_deg", "quad_source")]:
                stages.pop(k)
            stages.update(st2)
            stages["quad"] = None
            choice["selected"] = "no_quad"
        else:
            choice["selected"] = "pagenet_quad"
        stages["quad_alt"] = choice
    Hc, size, I1, S1, sheet, page_src, model, n = (hyp[k] for k in ("Hc", "size", "I1", "S1", "sheet", "page_src",
                                                                   "model", "n"))
    bm = BackwardMap.from_homography(Hc, size)
    cur = I1
    paper_src = keep_pts = None
    if n >= 20:
        stages["surface_candidates"] = model.fit_candidates
        stages["surface_rms_px"] = [model.rms0, model.rms]
        stages["surface_err_px"] = [model.err_median, model.err_p90]
        stages["ruling_deg"] = float(np.degrees(model.p[6]))
        mask = page_mask(img) if cfg.full_page else None
        if cfg.page_seg_extent and sheet is not None:
            mask = sheet
        good = ()
        # content structures (staves, rows) are surely on the sheet; free line
        # fragments and strokes may be the paper border itself
        on_sheet = model.obs[model.content]
        if cfg.full_page and cfg.edges and not (cfg.page_seg_extent and sheet is not None):
            mask = refine_page_mask(img, mask, on_sheet)
        if cfg.full_page and cfg.edges and mask is not None:
            # the mask must hold the content, else it is a lit patch / shadow
            # outline rather than the sheet
            qi = np.round(on_sheet).astype(int)
            okq = (qi[:, 0] >= 0) & (qi[:, 0] < W0) & (qi[:, 1] >= 0) & (qi[:, 1] < H0)
            inside = np.zeros(len(qi), bool)
            inside[okq] = mask[qi[okq, 1], qi[okq, 0]] > 0
            stages["mask_holds_structures"] = float(inside.mean())
            cand = page_edges(mask, Hc, img=img) if inside.mean() >= 0.97 else {}
            sel = model.select_edges(cand, Hc, tol=cfg.edge_tol) if cand else {}
            stages["edges_candidates"] = {k: len(v) for k, v in cand.items()}
            stages["edges_ransac"] = getattr(model, "edge_info", {})
            if sel and cfg.edge_mode == "extent":
                good = tuple(sel)
            elif sel and model.add_edges(sel, Hc, w=cfg.edge_w, widen=cfg.edge_widen):
                model.fit(**{**model.fit_kw, "max_nfev": 100}, warm=True)
                err = model.edge_errors()
                stages["edges_err_px"] = err
                good = tuple(k for k, e in err.items() if e < cfg.edge_max_err)
                stages["surface_err_px"] = [model.err_median, model.err_p90]
        stages["edges_used"] = list(good)
        # sheet mask for the reading copy: only when it holds the content
        if mask is not None and stages.get("mask_holds_structures", 0) >= 0.97:
            paper_src = mask
        elif cfg.main_page and sheet is not None and stages.get("page_seg", {}).get("holds", 0) >= 0.97:
            paper_src = sheet   # no extent mask (crop_content): the learned sheet whitens the main page's rim
        extent = (model.page_extent(mask, edges=good, fold_check=cfg.extent_fold) if cfg.full_page
                  else model.content_extent())
        pts = None
        if cfg.ink_extent:
            extent, pts = grow_to_content(model, img, extent, S1.spacing_est,
                                          sheet=sheet if cfg.sheet_content else None)
        bm2 = model.backward_map(extent)
        W2, H2 = bm2.out_size
        finite = np.isfinite(bm2.grid).all(-1).mean()
        if 0.3 < W2 / size[0] < 3 and 0.2 < H2 / size[1] < 4 and finite > 0.9:
            bm = bm2
            cur = bm.apply(img, value=255)
            keep_pts = pts
        else:
            stages["surface_rejected"] = True
    else:
        stages["surface_rejected"] = True
    T["surface"] = time.time() - t

    field_kw = cfg.field_kw
    if n >= 20 and len(getattr(model, "staff_groups", ())) > 1:
        # staves of more than one printed size on the photo (violin above
        # piano): the field keeps the sizes apart as well. Decided here, on the
        # photo, not on the surface output, where a badly stretched page shows
        # one size at two spacings (dev 0091 / 0111 / 0115).
        field_kw = {**(field_kw or {}), "size_classes": True}
        stages["staff_sizes"] = len(model.staff_groups)
    legacy = Config(use_field=cfg.residual == "field", use_line_mesh=cfg.residual == "mesh",
                    field_iters=cfg.field_iters, field_kw=field_kw,
                    field_kw_type=cfg.field_type)
    if cfg.residual == "mesh":
        legacy = replace(legacy, use_field=True)  # field remains the fallback for non-staff pages
    if cfg.residual == "none":
        legacy = replace(legacy, use_field=False)
    res = _finish(img, cur, bm, legacy, stages, T, t0, verbose, local_lines=cfg.local_lines,
                  merge_rows=cfg.merge_rows, keep_pts=keep_pts if cfg.ink_extent else None,
                  margin=cfg.margin if cfg.ink_extent else 0.0, stem_refine=cfg.stem_refine,
                  seg_src=seg_src if "c" in cfg.seg_stages else None,
                  seg_final=seg_src if "f" in cfg.seg_stages else None,
                  seg_parts=cfg.seg_parts, stem_bridge=cfg.stem_bridge, page_src=page_src, seg_fuse=cfg.seg_fuse,
                  page_rim=cfg.page_rim, jp_rows=cfg.jp_rows, glyph=cfg.glyph)
    if cfg.rectify_margins:
        res = rectify_margins(img, res, cfg)
    ms = margin_stats(res.structures)
    if ms is not None:   # diagnostics: slant of the staves' left / right ends, width change top -> bottom
        res.stages["margins"] = {"left_deg": ms["left"][2], "right_deg": ms["right"][2], "keystone": ms["keystone"]}
    if cfg.output_res == "native":
        res = to_native(img, res)
    if paper_src is not None:
        res.paper = (res.bmap.apply(paper_src, interp=cv2.INTER_LINEAR, value=0) > 127).astype(np.uint8) * 255
    return _main_page(res, cfg, sheet)


def _main_page(res, cfg, sheet=None):
    if not cfg.main_page:
        return res
    from .mainpage import main_page
    t = time.time()
    res = main_page(res, sheet)
    res.timings["main_page"] = time.time() - t
    return res


def to_native(img, res, tol=0.02):
    """Resample the final output at the photo's resolution (one sampling of ``img``).

    Detection and fitting run on a ~1600 px working canvas; a 12 MP photo
    holds about twice that linear resolution on the page. The final map is
    rescaled so that an output pixel covers ~one photo pixel (median over the
    content, ``native_scale``) and the photo is sampled once more from the
    original; structures / content points are scaled to the new canvas. Small
    photos get a smaller output (the canvas would only upsample them)."""
    t = time.time()
    s = native_scale(res.bmap, img.shape[1::-1], res.content)
    res.stages["output_scale"] = round(s, 4)
    if abs(s - 1) < tol:
        return res
    bm = res.bmap.rescale(s)
    image = bm.apply(img, value=255)
    norm = normalize_illumination(to_gray(image))
    content = None if res.content is None else np.asarray(res.content, float) * s
    T = dict(res.timings)
    T["native"] = time.time() - t
    T["total"] = T.get("total", 0.0) + T["native"]
    return replace(res, image=image, enhanced=norm, bmap=bm, structures=scale_structures(res.structures, s),
                   content=content, timings=T)


def rectify_margins(img, res, cfg):
    """Map the text block onto a rectangle (Stamatopoulos et al. 2011, the
    boundary terms of Jiang et al. CVPR 2022): fit lines through the staves'
    left ends and right ends of the corrected page and stretch every row
    linearly so that both become vertical, at the width of the middle row.
    A justified score has its clefs and final barlines on two vertical lines;
    a residual slant (shear) or a width change from top to bottom (keystone)
    left by the surface + field shows there first. Skipped when the margins
    rest on few rows or the correction would be large (a misread margin)."""
    ms = margin_stats(res.structures, min_rows=cfg.rect_min_rows)
    st = res.stages
    if ms is None:
        st["rectify_margins"] = "skipped: few rows"
        return res
    if max(abs(ms["left"][2]), abs(ms["right"][2])) > cfg.rect_max_deg or abs(ms["keystone"]) > cfg.rect_max_kst:
        st["rectify_margins"] = "skipped: large"
        return res
    (kL, cL), (kR, cR) = ms["lines"]["left"], ms["lines"]["right"]
    ym = 0.5 * sum(ms["y"])
    Lm, Rm = cL + kL * ym, cR + kR * ym

    def back(q):   # rectified output -> current output
        L, R = cL + kL * q[:, 1], cR + kR * q[:, 1]
        return np.stack([L + (q[:, 0] - Lm) * (R - L) / (Rm - Lm), q[:, 1]], 1)

    bm = res.bmap.compose(BackwardMap.from_function(back, res.bmap.out_size, stride=res.bmap.stride))
    image = bm.apply(img, value=255)
    content = res.content
    if content is not None and len(content):
        c = np.asarray(content, float)
        L, R = cL + kL * c[:, 1], cR + kR * c[:, 1]
        content = np.stack([Lm + (c[:, 0] - L) * (Rm - Lm) / (R - L), c[:, 1]], 1)
    norm, _, S = _detect(image, merge_rows=cfg.merge_rows, stem_refine=cfg.stem_refine)
    st["rectify_margins"] = {"left_deg": ms["left"][2], "right_deg": ms["right"][2], "keystone": ms["keystone"]}
    return replace(res, image=image, enhanced=norm, bmap=bm, structures=S,
                   page_type=S.page_type, content=content)


# Small detection models only (no learned warping); 2026-11-19 dev + real-photo
# evaluation in reports/model_route_20261119.md.
#   page: U²-Netp sheet mask -> evidence only on the sheet, page quad when the
#         brightness quad fails (fitted with and without it, the more
#         rectangular sheet outline wins), both ruling starts finished and the
#         most rectangular outline kept, extent stops at surface folds.
#   all:  page + homr staff lines fused per staff (stems stay classical).
#         The residual field also keeps the sheet's own paper edges straight
#         (field sheet_w, page.sheet_sides): the margins outside the outermost
#         staves have no other evidence and kept the surface's error (IMG_3683
#         fan curl: left paper edge slanted / bent). Dev 2026-11-19 vs page
#         without it: edge_deg 53 better / 13 worse, lalign 53 / 4, kst 50 / 20,
#         ad_c 31 / 26, ld_c 31 / 33.
MODEL_PRESETS = {
    "off": {},
    "page": {"page_seg": "u2netp", "page_quad": True, "quad_alt": True, "page_erode": 0.002, "b_select": "outline",
             "extent_fold": True, "field_kw": {**DEFAULT_FIELD, "sheet_w": 4.0}},
}
MODEL_PRESETS["all"] = {**MODEL_PRESETS["page"], "seg": "homr", "seg_parts": "staff", "seg_fuse": "staff"}


def preset_config(name="off", **kw) -> MetricConfig:
    return MetricConfig(**{**MODEL_PRESETS[name], **kw})


BENCH_METHODS = {
    # the bench measures geometry through the map; keep the working canvas so
    # runs stay comparable with earlier ones (and fast)
    "metric": lambda img: dewarp_metric(img, MetricConfig(output_res="canvas")),          # the default pipeline
    "gcs": lambda img: dewarp_metric(img, MetricConfig(residual="none", output_res="canvas")),  # surface only
    "models_page": lambda img: dewarp_metric(img, preset_config("page", output_res="canvas")),
    "models_all": lambda img: dewarp_metric(img, preset_config("all", output_res="canvas")),
}


def method_from_spec(spec: str):
    """'gcs:residual=field,focal_scale=0.8,fit.lam_curv=20,c.lam_v=0.3,w.vseg=3' -> callable(img).

    ``c.*`` / ``w.*`` override single entries of the tuned field / weight defaults,
    ``j.*`` / ``m.*`` the field on jianpu / mixed pages only."""
    _, _, args = spec.partition(":")
    kw, fit = {"field_kw": dict(DEFAULT_FIELD), "weights": dict(DEFAULT_WEIGHTS),
               "field_type": {k: dict(v) for k, v in DEFAULT_FIELD_TYPE.items()}}, {}
    for item in filter(None, args.split(",")):
        k, v = item.split("=")
        try:
            v = float(v) if "." in v or v.replace("-", "").isdigit() else {"true": True, "false": False}.get(v, v)
        except ValueError:
            pass
        if k.startswith("fit."):
            fit[k[4:]] = v
        elif k.startswith("c."):
            kw["field_kw"][k[2:]] = v
        elif k.startswith("w."):
            kw["weights"][k[2:]] = v
        elif k.startswith("j."):
            kw["field_type"]["jianpu"][k[2:]] = v
        elif k.startswith("m."):
            kw["field_type"]["mixed"][k[2:]] = v
        else:
            kw[k] = v
    if "field_iters" in kw:
        kw["field_iters"] = int(kw["field_iters"])
    kw.setdefault("output_res", "canvas")   # bench: see BENCH_METHODS
    cfg = MetricConfig(**kw, fit_kw=fit or None)
    return lambda img: dewarp_metric(img, cfg)
