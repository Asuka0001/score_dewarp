"""Real-photo evaluation on CPMS: notehead <-> staff-line relation ("符头与谱线关系").

CPMS labels every notehead with its staff position (half-spacing steps above
the middle line; G4 = -2, B4 = 0 on a treble staff) on a downscaled copy of the
photo whose scale is not given and differs per photo. The scale is calibrated
per photo: the one that puts the most label points on noteheads (learned
notehead mask), then each point is snapped to its notehead blob centre.

Two evaluations, each for the classical detector and for --seg homr:
  detect  staves detected on the coarse page (photo -> page homography);
          each notehead is placed on the detected staff it lies in and its
          position read from the five detected lines -> step accuracy, coverage
  e2e     the full correction; noteheads mapped to the output with the
          forward map, the output judged by ONE fixed referee (the classical
          detector) for both methods -> step accuracy, staff-line flatness

    python scripts/eval_cpms.py --out outputs/cpms_eval [--limit N] [--workers 4]
"""
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
from scoredewarp.core.image_input import read_photo  # noqa: E402
from scoredewarp.core.learned import segment  # noqa: E402
from scoredewarp.core.metric_pipeline import MetricConfig, coarse, detect_at_scale, dewarp_metric  # noqa: E402
from scoredewarp.core.pipeline import _detect  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CPMS = sorted(glob.glob(str(ROOT / "data/external/cpms/source/CPMS-*")))


def calibrate(P, head):
    """Label coords P (top-left of notehead boxes) -> photo notehead centres."""
    H, W = head.shape
    dil = cv2.dilate(head, np.ones((5, 5), np.uint8))
    best = (-1, None)
    for s in np.arange(0.30, 0.60, 0.0025):
        q = np.round((P + 6) / s).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
        sc = dil[q[ok, 1], q[ok, 0]].sum()
        if sc > best[0]:
            best = (sc, s)
    s = best[1]
    n, lab, stats, cent = cv2.connectedComponentsWithStats(head, connectivity=8)
    Q = (P + 6) / s
    out = Q.copy()
    snapped = np.zeros(len(Q), bool)
    r = int(round(4 / s))
    for i, (x, y) in enumerate(np.round(Q).astype(int)):
        y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, H), max(x - r, 0), min(x + r + 1, W)
        ids = np.unique(lab[y0:y1, x0:x1])
        ids = ids[ids > 0]
        if len(ids):
            j = ids[np.argmin(np.hypot(*(cent[ids] - (x, y)).T))]
            if stats[j, cv2.CC_STAT_AREA] < (12 / s) ** 2:     # one notehead, not a chord / beam blob
                out[i] = cent[j]
                snapped[i] = True
    return out, s, float(best[0] / len(P)), snapped


def staff_lines(S):
    """Per staff: x samples and the five line ys (from the 5-line tuples)."""
    out = []
    for st in S.staves:
        T = np.array(st.tuples)                     # (n, 5, 2)
        if len(T) < 2:
            continue
        o = np.argsort(T[:, 0, 0])
        out.append((T[o, 0, 0], T[o, :, 1], st.spacing))
    return out


def positions(pts, lines):
    """Notehead points -> (staff index or -1, step position) on detected staves."""
    res = []
    for x, y in pts:
        best = (np.inf, -1, 0)
        for k, (xs, ys, sp) in enumerate(lines):
            if not (xs[0] - 1.5 * sp <= x <= xs[-1] + 1.5 * sp):
                continue
            yl = np.array([np.interp(x, xs, ys[:, i]) for i in range(5)])
            spl = (yl[4] - yl[0]) / 4
            if spl <= 0:
                continue
            d = (yl[2] - y) / (spl / 2)
            if abs(d) <= 12 and abs(d) < best[0]:
                best = (abs(d), k, int(np.round(d)))
        res.append(best[1:])
    return res


def flatness(lines):
    """Staff lines on a rectified page: residual from a straight line / spacing, and slope (deg)."""
    res, tilt = [], []
    for xs, ys, sp in lines:
        if xs[-1] - xs[0] < 10 * sp:
            continue
        for i in range(5):
            c = np.polyfit(xs, ys[:, i], 1)
            res.append(np.std(ys[:, i] - np.polyval(c, xs)) / sp)
            tilt.append(abs(np.degrees(np.arctan(c[0]))))
    return (float(np.median(res)) if res else None, float(np.median(tilt)) if tilt else None)


def score(pos, labels):
    hit = np.array([k >= 0 for k, _ in pos])
    ok = np.array([k >= 0 and p == l for (k, p), l in zip(pos, labels)])
    return {"coverage": float(hit.mean()), "step_acc": float(ok.mean()),
            "step_acc_found": float(ok[hit].mean()) if hit.any() else None}


def run(path):
    name = Path(path).stem
    lab = json.load(open(Path(path).parent.parent / "label" / f"{name}.json"))
    P = np.array([[p["x"], p["y"]] for v in lab.values() for p in v], float)
    L = np.array([p["position"] for v in lab.values() for p in v])
    img = read_photo(path)
    seg = segment(img)
    pts, s, hit, snapped = calibrate(P, (seg[..., 2] > 127).astype(np.uint8))
    row = {"name": name, "n": len(P), "scale": s, "calib_hit": hit, "snapped": float(snapped.mean())}
    for mode in ("off", "homr"):
        cfg = MetricConfig(seg=mode)
        Hc, size, I1, *_ = coarse(img, cfg)
        Hc, size, I1, S = detect_at_scale(img, Hc, size, I1, cfg, {}, seg if mode == "homr" else None)
        q = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hc).reshape(-1, 2)
        row[f"detect_{mode}"] = {**score(positions(q, staff_lines(S)), L), "staves": len(S.staves)}
        res = dewarp_metric(img, cfg)
        qo = res.forward_points(pts)
        # fixed referees, the same for both methods' outputs
        for ref in ("classical", "homr"):
            rs = segment(res.image) if ref == "homr" else None
            _, _, R = _detect(res.image, robust=False, local_lines=True, stem_refine=True, seg=rs)
            lines = staff_lines(R)
            fl, tl = flatness(lines)
            row[f"e2e_{mode}_ref{ref}"] = {**score(positions(qo, lines), L), "staves": len(R.staves),
                                           "flat": fl, "tilt_deg": tl, "out_size": list(res.image.shape[1::-1])}
        row[f"_res_{mode}"] = res
    return row


OUT = None


def _init(out):
    global OUT
    OUT = Path(out)


def worker(path):
    try:
        row = run(path)
    except Exception as e:  # keep the batch going
        return {"name": Path(path).stem, "error": repr(e)}
    for mode in ("off", "homr"):
        res = row.pop(f"_res_{mode}")
        cv2.imwrite(str(OUT / "rectified" / f"{row['name']}_{mode}.jpg"), res.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps({k: v for k, v in row.items() if k.startswith(("name", "detect", "e2e"))}), flush=True)
    return row


def summarize(rows):
    rows = [r for r in rows if "error" not in r]
    lines = [f"# CPMS notehead–staff evaluation ({len(rows)} photos, "
             f"{sum(r['n'] for r in rows)} noteheads)", "",
             "Step accuracy: notehead placed on the right line / space (half-spacing step), "
             "over all labelled noteheads (a notehead outside every detected staff counts as wrong). "
             "Medians over photos; pooled = over all noteheads.", "",
             "Stages: detect = detector on the coarse photo page; e2e = after full correction, judged by a fixed "
             "referee detector (classical, or with the learned masks) applied identically to both outputs. "
             "flat = staff-line residual std from a straight line / spacing; tilt = staff-line slope.", "",
             "| stage | method | step acc (median) | step acc (pooled) | coverage | acc on covered | staves | flat (/sp) | tilt (°) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for st in ("detect", "e2e_refclassical", "e2e_refhomr"):
        for m in ("off", "homr"):
            k = f"{st}_{m}" if st == "detect" else f"e2e_{m}_{st[4:]}"
            g = lambda f: [r[k][f] for r in rows if r[k].get(f) is not None]
            pooled = sum(r[k]["step_acc"] * r["n"] for r in rows) / sum(r["n"] for r in rows)
            fl = f"{np.median(g('flat')):.3f}" if g("flat") else "—"
            tl = f"{np.median(g('tilt_deg')):.2f}" if g("tilt_deg") else "—"
            lines.append(f"| {st} | {'classical' if m == 'off' else 'seg homr'} | {np.median(g('step_acc')):.3f} | "
                         f"{pooled:.3f} | {np.median(g('coverage')):.3f} | {np.median(g('step_acc_found')):.3f} | "
                         f"{np.median(g('staves')):.0f} | {fl} | {tl} |")
    lines += ["", "Paired per photo (homr − classical), step accuracy:", ""]
    for st, a, b in (("detect", "detect_off", "detect_homr"),
                     ("e2e, classical referee", "e2e_off_refclassical", "e2e_homr_refclassical"),
                     ("e2e, homr referee", "e2e_off_refhomr", "e2e_homr_refhomr")):
        d = np.array([r[b]["step_acc"] - r[a]["step_acc"] for r in rows])
        lines.append(f"- {st}: better {(d > 1e-9).sum()}, worse {(d < -1e-9).sum()}, same {(np.abs(d) <= 1e-9).sum()}, "
                     f"mean {d.mean():+.4f}")
    lines += ["", f"Calibration: median label-on-notehead hit rate {np.median([r['calib_hit'] for r in rows]):.3f}, "
              f"snapped to a single notehead blob {np.median([r['snapped'] for r in rows]):.3f}."]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/cpms_eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT = args.out
    (OUT / "rectified").mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(f"{CPMS[0]}/data/*"))[: args.limit or None]
    with Pool(args.workers, initializer=_init, initargs=(str(OUT),)) as pool:
        rows = pool.map(worker, paths, chunksize=1)
    with open(OUT / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rep = summarize(rows)
    (OUT / "summary.md").write_text(rep)
    print(rep)
    errs = [r for r in rows if "error" in r]
    if errs:
        print("errors:", errs)
