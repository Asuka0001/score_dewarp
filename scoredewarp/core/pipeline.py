"""End-to-end score photo dewarping.

    A  coarse     : page quad -> homography, projection-profile deskew
    B  surface3d  : structure-constrained 3-D page + camera fit (global curl / perspective)
    C  field      : non-rigid residual field from re-detected structures (local bumps)
    D  outputs    : rectified colour image, enhanced (illumination-normalised)
                    image, score-type label, regions, keypoints, backward map
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .field import fit_field
from .page import detect_page_quad, estimate_skew, quad_homography, rotation_homography
from .preprocess import binarize, normalize_illumination, to_gray
from .structures import Structures, detect_structures
from .surface3d import SurfaceModel
from .transforms import BackwardMap


@dataclass
class Config:
    use_page_quad: bool = True
    use_surface: bool = True
    use_field: bool = True
    field_priors: tuple = ("staff", "jianpu", "text", "vertical")
    surface_priors: tuple = ("staff", "jianpu", "text", "vertical")
    coarse_width: int = 1600
    test_rot90: bool = False
    field_node: int = 48
    field_iters: int = 2


@dataclass
class Result:
    image: np.ndarray
    enhanced: np.ndarray
    bmap: BackwardMap                 # output -> input photo
    structures: Structures            # detected on the final output
    page_type: str
    stages: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def forward_points(self, pts):
        return self.bmap.forward_points(pts)


def _detect(img, *, robust=True):
    norm = normalize_illumination(to_gray(img))
    valid = (to_gray(img) > 0).astype(np.uint8) if img.ndim == 2 else None
    ink = binarize(norm)
    return norm, ink, detect_structures(norm, ink, robust=robust)


def _filter_structures(S: Structures, priors):
    """Ablation helper: drop structure types not in `priors`."""
    import copy
    S2 = copy.copy(S)
    from .structures import Fragment
    S2.fragments = list(S.fragments)
    if "staff" not in priors:   # staff lines degrade to generic horizontal lines
        S2.staves = []
        S2.fragments = [Fragment(f.pts) for f in S.fragments]
    if "jianpu" not in priors:
        S2.rows = [r for r in S2.rows if r.kind != "jianpu"]
        if "text" not in priors:  # drop free (non-staff) horizontal lines, keep indices stable
            S2.fragments = [f if f.staff >= 0 else Fragment(f.pts[:1]) for f in S2.fragments]
    if "text" not in priors:
        S2.rows = [r for r in S2.rows if r.kind != "text"]
    if "vertical" not in priors:
        S2.vsegs = []
    return S2


def dewarp(img: np.ndarray, cfg: Config | None = None, verbose=False) -> Result:
    cfg = cfg or Config()
    T = {}
    t0 = time.time()
    H0, W0 = img.shape[:2]
    stages = {}

    # ---- A: coarse -----------------------------------------------------------
    quad = detect_page_quad(img) if cfg.use_page_quad else None
    if quad is not None:
        Hq, size = quad_homography(quad, cfg.coarse_width)
    else:
        sc = cfg.coarse_width / W0
        Hq, size = np.diag([sc, sc, 1.0]), (int(W0 * sc), int(H0 * sc))
    I1 = cv2.warpPerspective(img, Hq, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    norm1 = normalize_illumination(to_gray(I1))
    ang, r90 = estimate_skew(binarize(norm1), test_rot90=cfg.test_rot90)
    R, size = rotation_homography(ang, size, r90)
    Hc = R @ Hq
    I1 = cv2.warpPerspective(img, Hc, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    bm = BackwardMap.from_homography(Hc, size)
    stages["quad"] = None if quad is None else quad.tolist()
    stages["skew_deg"] = float(ang)
    T["coarse"] = time.time() - t0

    # ---- B: 3-D surface --------------------------------------------------------
    cur = I1
    if cfg.use_surface:
        t = time.time()
        # Keep the fitted geometry's established evidence distribution. Faint
        # line recovery is enabled for final exported structures below.
        _, _, S1 = _detect(I1, robust=False)
        S1f = _filter_structures(S1, cfg.surface_priors)
        sm = SurfaceModel((W0, H0))
        n = sm.build(S1f, Hc)
        if n >= 20:
            sm.fit()
            stages["surface_rms_px"] = [sm.rms0, sm.rms]
            stages["surface_coeffs"] = sm.p[6:14].tolist()
            bm2 = sm.backward_map(out_scale=1.0)
            # sanity check: reject degenerate fits
            W2, H2 = bm2.out_size
            if 0.3 < W2 / size[0] < 3 and 0.2 < H2 / size[1] < 4 and np.isfinite(bm2.grid).all():
                bm = bm2
                cur = bm.apply(img, value=255)
            else:
                stages["surface_rejected"] = True
        stages["structures_coarse"] = S1.summary()
        T["surface"] = time.time() - t

    return _finish(img, cur, bm, cfg, stages, T, t0, verbose)


def refine_dewarp(img: np.ndarray, initial_map: BackwardMap, cfg: Config | None = None,
                  verbose=False) -> Result:
    """Refine an existing learned rectification with stage C, then detect output.

    A/B are skipped: the nonlinear learned map cannot initialize B's homography
    interface. All residual maps compose back to the original photo, so each
    candidate output is sampled directly from the original pixels.
    """
    cfg = cfg or Config()
    t0 = time.time()
    cur = initial_map.apply(img, value=255)
    return _finish(img, cur, initial_map, cfg, {"initialization": "network", "surface_skipped": True},
                   {}, t0, verbose)


def _finish(img, cur, bm, cfg, stages, T, t0, verbose):
    # ---- C: residual field ---------------------------------------------------
    if cfg.use_field:
        t = time.time()
        for _ in range(cfg.field_iters):
            _, _, S2 = _detect(cur, robust=False)
            S2f = _filter_structures(S2, cfg.field_priors)
            fb = fit_field(S2f, node=cfg.field_node, priors=cfg.field_priors)
            if fb is None:
                break
            bm = bm.compose(fb)
            cur = bm.apply(img, value=255)
        T["field"] = time.time() - t

    # ---- D: outputs ------------------------------------------------------------
    t = time.time()
    norm, ink, Sf = _detect(cur)
    enhanced = norm
    T["final_detect"] = time.time() - t
    T["total"] = time.time() - t0
    if verbose:
        print({k: round(v, 2) for k, v in T.items()}, Sf.summary())
    return Result(cur, enhanced, bm, Sf, Sf.page_type, stages, T)


def result_json(res: Result):
    S = res.structures
    out = {
        "page_type": res.page_type,
        "page_type_source": res.stages.get("kind_decision", {}).get("source", "structure_detector"),
        "summary": S.summary(),
        "regions": [{"type": t, "polygon": np.round(p, 1).tolist()} for t, p in S.regions],
        "staves": [{"spacing": st.spacing,
                    "lines": [np.round(S.fragments[fi].pts, 1).tolist() for fi in st.frag_ids],
                    "line_index": [S.fragments[fi].line for fi in st.frag_ids]} for st in S.staves],
        "jianpu_rows": [{"height": r.height,
                         "glyphs": [[b.x0, b.y0, b.x1, b.y1, b.digit] for b in r.blobs]}
                        for r in S.rows if r.kind == "jianpu"],
        "text_rows": [{"height": r.height, "boxes": [[b.x0, b.y0, b.x1, b.y1] for b in r.blobs]}
                      for r in S.rows if r.kind == "text"],
        "stages": res.stages,
        "timings_s": {k: round(v, 3) for k, v in res.timings.items()},
        "output_size": list(res.bmap.out_size),
    }
    return out
