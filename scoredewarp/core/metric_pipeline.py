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

from .gcs import GCSModel
from .page import (detect_page_quad, estimate_skew, page_edges, page_mask, quad_homography, refine_page_mask,
                   rotation_homography)
from .pipeline import Config, _detect, _finish
from .preprocess import binarize, normalize_illumination, to_gray
from .transforms import BackwardMap


# Tuned on the Doc3D-retextured dev split (data/bench/doc3d_score/dev), see
# reports/metric_route_20261117.md. Vertical strokes need a high weight: they are
# the only evidence against shear, and horizontal lines outnumber them. The
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
DEFAULT_FIELD = {"tuple_u": 0.0, "vseg_w": 4.0, "lam_u": 3.0, "lam_v": 0.5, "spacing_gate": 9.0, "ends_w": 2.0,
                 "keep_scale": True, "anchor": True}
DEFAULT_FIELD_TYPE = {"jianpu": {"vseg_w": 8.0}, "mixed": {"lam_u": 1.5}}   # on top of DEFAULT_FIELD


@dataclass
class MetricConfig:
    coarse_width: int = 1600
    robust_detect: bool = False
    local_lines: bool = True
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
    residual: str = "field"          # none | field | mesh
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


def coarse(img, cfg):
    quad = detect_page_quad(img)
    Hq, size = _coarse_homography(img, quad, cfg.coarse_width)
    I1 = cv2.warpPerspective(img, Hq, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ang, r90 = estimate_skew(binarize(normalize_illumination(to_gray(I1))))
    R, size = rotation_homography(ang, size, r90)
    Hc = R @ Hq
    I1 = cv2.warpPerspective(img, Hc, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Hc, size, I1, quad, ang


def detect_at_scale(img, Hc, size, I1, cfg, stages):
    """Detect structures; redo at a larger coarse scale for small staff lines.

    The detector works in integer pixels (run lengths >= 6 px). Dense scores
    photographed small (e.g. orchestral pages) fall below that, so when staves
    are found with spacing < 9 px the coarse image is re-rendered with spacing
    ~12 px. Only measured staff spacing triggers this: other periodic ink
    (digit strokes) would upscale jianpu pages for nothing, which hurts them.
    """
    _, _, S = _detect(I1, robust=cfg.robust_detect, local_lines=cfg.local_lines, merge_rows=cfg.merge_rows)
    if cfg.adaptive_scale != "off" and S.staves:
        sp = float(np.median([st.spacing for st in S.staves]))
        if sp < 9:
            f = min(12.0 / sp, 2.5)
            Hc = np.diag([f, f, 1.0]) @ Hc
            size = (int(round(size[0] * f)), int(round(size[1] * f)))
            I1 = cv2.warpPerspective(img, Hc, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            _, _, S = _detect(I1, robust=cfg.robust_detect, local_lines=cfg.local_lines, merge_rows=cfg.merge_rows)
            stages["coarse_upscale"] = f
    return Hc, size, I1, S


def dewarp_metric(img: np.ndarray, cfg: MetricConfig | None = None, verbose=False):
    cfg = cfg or MetricConfig()
    t0 = time.time()
    T, stages = {}, {"model": "gcs"}
    H0, W0 = img.shape[:2]
    Hc, size, I1, quad, ang = coarse(img, cfg)
    bm = BackwardMap.from_homography(Hc, size)
    cur = I1
    stages["quad"] = None if quad is None else quad.tolist()
    stages["skew_deg"] = float(ang)
    T["coarse"] = time.time() - t0

    t = time.time()
    Hc, size, I1, S1 = detect_at_scale(img, Hc, size, I1, cfg, stages)
    bm = BackwardMap.from_homography(Hc, size)
    cur = I1
    stages["structures_coarse"] = S1.summary()
    paper_src = None
    model = GCSModel((W0, H0), focal=cfg.focal_scale * max(W0, H0))
    n = model.build(S1, Hc, weights=cfg.weights)
    stages["surface_observations"] = int(n)
    if n >= 20:
        model.fit(**(cfg.fit_kw or {}))
        stages["surface_rms_px"] = [model.rms0, model.rms]
        stages["surface_err_px"] = [model.err_median, model.err_p90]
        stages["ruling_deg"] = float(np.degrees(model.p[6]))
        mask = page_mask(img) if cfg.full_page else None
        good = ()
        # content structures (staves, rows) are surely on the sheet; free line
        # fragments and strokes may be the paper border itself
        on_sheet = model.obs[model.content]
        if cfg.full_page and cfg.edges:
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
        extent = model.page_extent(mask, edges=good) if cfg.full_page else model.content_extent()
        bm2 = model.backward_map(extent)
        W2, H2 = bm2.out_size
        finite = np.isfinite(bm2.grid).all(-1).mean()
        if 0.3 < W2 / size[0] < 3 and 0.2 < H2 / size[1] < 4 and finite > 0.9:
            bm = bm2
            cur = bm.apply(img, value=255)
        else:
            stages["surface_rejected"] = True
    else:
        stages["surface_rejected"] = True
    T["surface"] = time.time() - t

    legacy = Config(use_field=cfg.residual == "field", use_line_mesh=cfg.residual == "mesh",
                    field_iters=cfg.field_iters, field_kw=cfg.field_kw,
                    field_kw_type=cfg.field_type)
    if cfg.residual == "mesh":
        legacy = replace(legacy, use_field=True)  # field remains the fallback for non-staff pages
    if cfg.residual == "none":
        legacy = replace(legacy, use_field=False)
    res = _finish(img, cur, bm, legacy, stages, T, t0, verbose, local_lines=cfg.local_lines,
                  merge_rows=cfg.merge_rows)
    if paper_src is not None:
        res.paper = (res.bmap.apply(paper_src, interp=cv2.INTER_LINEAR, value=0) > 127).astype(np.uint8) * 255
    return res


BENCH_METHODS = {
    "metric": lambda img: dewarp_metric(img, MetricConfig()),                 # the default pipeline
    "gcs": lambda img: dewarp_metric(img, MetricConfig(residual="none")),     # surface only
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
    cfg = MetricConfig(**kw, fit_kw=fit or None)
    return lambda img: dewarp_metric(img, cfg)
