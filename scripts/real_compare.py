"""Before / after comparison on real photos (CPMS + own photos), no flat GT.

Per photo and config (both with the ``page`` model preset):
  edge_bend   median distance of the staff rows' left / right ends from a robust
              (Theil-Sen) straight line through them, on the output (staff spacings; 0 = the text
              block's sides are straight)
  edge_slant  |angle| of those lines from vertical (deg)
  step_acc    CPMS only: notehead line / space read on the output by the fixed
              classical referee (``eval_cpms``), vs the labels
Side-by-side images with the detected row ends (red) and the fitted lines (green).

  python scripts/real_compare.py --out outputs/real_compare_20261119 --workers 6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

ROOT = Path(__file__).resolve().parents[1]
OLD_FIELD = dict(h_ruler="vertical", row_u=0.3, text_u=None, block_jp=False, edge_w=0.0)
OUT = None


def configs():
    from scoredewarp.core.metric_pipeline import DEFAULT_FIELD, preset_config
    return {"before": preset_config("page", output_res="canvas", field_kw={**DEFAULT_FIELD, **OLD_FIELD}),
            "now": preset_config("page", output_res="canvas")}


def edges(S):
    from scipy.stats import theilslopes
    from scoredewarp.core.field import _row_ends
    if not S.staves:
        return None, []
    sp = float(np.median([s.spacing for s in S.staves]))
    out, lines = {}, []
    for side, name in ((0, "left"), (-1, "right")):
        P = _row_ends(S, side, reject=1e9)
        if len(P) < 3:
            continue
        k, c = theilslopes(P[:, 0], P[:, 1])[:2]   # robust: the referee misreads a row end now and then
        r = P[:, 0] - (c + k * P[:, 1])
        out[name] = (float(np.median(np.abs(r)) / sp), float(abs(np.degrees(np.arctan(k)))))
        lines.append((P, k, c))
    if not out:
        return None, lines
    return {"edge_bend": float(np.mean([v[0] for v in out.values()])),
            "edge_slant": float(np.mean([v[1] for v in out.values()])), "rows": int(len(lines[0][0]))}, lines


def draw(img, lines, label, h=1100):
    im = img.copy()
    for P, k, c in lines:
        y0, y1 = P[:, 1].min() - 50, P[:, 1].max() + 50
        cv2.line(im, (int(c + k * y0), int(y0)), (int(c + k * y1), int(y1)), (0, 170, 0), 3, cv2.LINE_AA)
        for x, y in P:
            cv2.circle(im, (int(x), int(y)), 9, (0, 0, 230), -1, cv2.LINE_AA)
    im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA)
    bar = np.full((50, im.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, im])


def run(item):
    src, path = item
    from scoredewarp.core.image_input import read_photo
    from scoredewarp.core.metric_pipeline import dewarp_metric
    from scoredewarp.core.pipeline import _detect
    name = Path(path).stem
    row = {"set": src, "name": name}
    try:
        img = read_photo(path)
        cpms = None
        if src == "cpms":
            import eval_cpms as ec
            from scoredewarp.core.learned import segment
            lab = json.load(open(Path(path).parent.parent / "label" / f"{name}.json"))
            P = np.array([[p["x"], p["y"]] for v in lab.values() for p in v], float)
            L = np.array([p["position"] for v in lab.values() for p in v])
            pts, *_ = ec.calibrate(P, (segment(img)[..., 2] > 127).astype(np.uint8))
            cpms = (pts, L, ec)
        panels = []
        for cname, cfg in configs().items():
            res = dewarp_metric(img, cfg)
            _, _, R = _detect(res.image, robust=False, local_lines=True, stem_refine=True)
            e, lines = edges(R)
            r = dict(e or {})
            if cpms:
                pts, L, ec = cpms
                r["step_acc"] = ec.score(ec.positions(res.forward_points(pts), ec.staff_lines(R)), L)["step_acc"]
            row[cname] = r
            lab = cname + (f"  bend {r['edge_bend']:.2f} sp  slant {r['edge_slant']:.2f} deg" if e else "  (no margins)")
            panels.append(draw(res.image, lines, lab))
        w = max(p.shape[1] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, 0, 0, w - p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                  for p in panels]
        gap = np.full((panels[0].shape[0], 24, 3), 255, np.uint8)
        cv2.imwrite(str(OUT / src / f"{name}.jpg"), np.hstack([panels[0], gap, panels[1]]),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    except Exception as ex:  # noqa: BLE001
        row["error"] = repr(ex)
    print(json.dumps(row), flush=True)
    return row


def _init(out):
    global OUT
    OUT = Path(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--own", default=str(ROOT / "data/manual_photos"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    for s in ("cpms", "own"):
        (args.out / s).mkdir(parents=True, exist_ok=True)
    cp = sorted(glob.glob(str(ROOT / "data/external/cpms/source/CPMS-*/data/*")))[: args.limit or None]
    own = sorted(p for p in glob.glob(f"{args.own}/*") if p.lower().endswith((".jpg", ".jpeg", ".png", ".heic")))
    items = [("own", p) for p in own] + [("cpms", p) for p in cp]
    with Pool(args.workers, initializer=_init, initargs=(str(args.out),)) as pool:
        rows = pool.map(run, items, chunksize=1)
    with open(args.out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
