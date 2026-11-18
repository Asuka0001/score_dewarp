"""Flat-page regressions and a fixed random CPMS subset for detection variants.

CPMS label calibration and both fixed referees follow eval_cpms.py. These are
development diagnostics, not an independent generalization benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import bench
from test_detection_route import SPECS, report
from eval_cpms import CPMS, calibrate, flatness, positions, score, staff_lines
from scoredewarp.core.image_input import read_photo
from scoredewarp.core.learned import segment
from scoredewarp.core.pipeline import _detect
from scoredewarp.core.transforms import mapping_quality


def run(path, names, kind, out):
    bench.register_new_methods()
    if os.environ.get("SCOREDEWARP_SEG_COREML") == "0":
        # Keep CPU inference from oversubscribing the concurrent geometry run.
        import onnxruntime as ort
        from scoredewarp.core import learned
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 3
        learned._SESSION = ort.InferenceSession(str(learned.model_path()), sess_options=options,
                                               providers=["CPUExecutionProvider"])
    path = Path(path)
    img = read_photo(path)
    rows = []
    if kind == "flat":
        scale = min(1., 1600 / max(img.shape[:2]))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]
        x, y = np.meshgrid(np.linspace(.05 * W, .95 * W, 40), np.linspace(.05 * H, .95 * H, 50))
        points = np.stack([x, y], -1)
        target, valid = points / W, np.ones(x.shape, bool)
    else:
        lab = json.loads((path.parent.parent / "label" / f"{path.stem}.json").read_text())
        P = np.array([[p["x"], p["y"]] for v in lab.values() for p in v], float)
        labels = np.array([p["position"] for v in lab.values() for p in v])
        seg = segment(img)
        points, scale, hit, snapped = calibrate(P, (seg[..., 2] > 127).astype(np.uint8))
    for name in names:
        start = time.time()
        result = bench.get_method(SPECS[name])(img)
        row = {"sample": f"{kind}/{path.parent.name if kind == 'flat' else path.stem}",
               "kind": path.parent.name.split("_")[0] if kind == "flat" else "staff",
               "method": name, "spec": SPECS[name], "time": time.time()-start,
               "notes": {"mapping_quality": mapping_quality(result.bmap, img.shape[1::-1])}}
        if kind == "flat":
            row.update(bench.geometry(points, target, valid, result.bmap, False, valid))
        else:
            q = result.forward_points(points)
            row.update(n=len(labels), calib_scale=scale, calib_hit=hit, snapped=float(snapped.mean()))
            for ref in ("classical", "homr"):
                rs = segment(result.image) if ref == "homr" else None
                _, _, S = _detect(result.image, robust=False, local_lines=True, stem_refine=True,
                                   seg=rs, seg_parts="staff+stem")
                lines = staff_lines(S)
                fl, tilt = flatness(lines)
                row[f"ref_{ref}"] = {**score(positions(q, lines), labels), "flat": fl, "tilt_deg": tilt}
        cv2.imwrite(str(Path(out) / f"{path.parent.name if kind == 'flat' else path.stem}_{name}.jpg"),
                    result.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=("flat", "cpms"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--methods", nargs="+", choices=SPECS, default=["classical", "homr", "stem_connected", "homr_rows"])
    args = ap.parse_args()
    if args.kind == "flat":
        paths = [Path(f"data/synth_dbg/{k}_000/flat.png") for k in ("staff", "jianpu", "mixed")]
    else:
        paths = sorted(Path(CPMS[0]).glob("data/*"))
        indices = np.random.default_rng(args.seed).choice(len(paths), min(args.limit, len(paths)), replace=False)
        paths = [paths[i] for i in sorted(indices)]
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "rows.jsonl").exists():
        ap.error("output already contains rows; choose a new directory")
    (args.out / "manifest.json").write_text(json.dumps({"kind": args.kind, "seed": args.seed,
        "paths": [str(p) for p in paths], "methods": {n: SPECS[n] for n in args.methods},
        "resolution": "canvas", "flat_alignment": "similarity, 5%-95% grid",
        "seg_cache": os.environ.get("SCOREDEWARP_SEG_CACHE"),
        "seg_coreml": os.environ.get("SCOREDEWARP_SEG_COREML", "1"),
        "cpu_session_threads": 1 if os.environ.get("SCOREDEWARP_SEG_COREML") == "0" else None,
        "referee": {"robust": False, "local_lines": True, "stem_refine": True, "seg_parts": "staff+stem"},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    rows = []
    with (args.out / "rows.jsonl").open("w") as log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, str(p), args.methods, args.kind, str(args.out)) for p in paths]
            for i, future in enumerate(as_completed(futures), 1):
                for row in future.result():
                    rows.append(row)
                    log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"{i}/{len(paths)}", flush=True)
    if args.kind == "flat":
        summary, _ = report(rows, args.methods)
        summary += "\nFlat controls use a similarity alignment, so aspect changes are penalized.\n"
        summary += "\n| sample | method | AD | LD |\n|---|---|---:|---:|\n"
        for row in sorted(rows, key=lambda r: (r["sample"], r["method"])):
            summary += f"| {row['sample']} | {row['method']} | {row['ad_c']:.4f} | {row['ld_c']:.4f} |\n"
    else:
        summary = (f"# CPMS development check: {len(paths)} photos, "
                   f"{sum(r['n'] for r in rows if r['method'] == args.methods[0])} noteheads\n\n"
                   f"Fixed random subset, seed {args.seed}. Labels calibrated using homr noteheads, "
                   "shared across methods. Canvas outputs; both referees fixed.\n\n"
                   "| method | referee | pooled step accuracy | photo median | photo median flat | photo median tilt |\n"
                   "|---|---|---:|---:|---:|---:|\n")
        for name in args.methods:
            group = [r for r in rows if r["method"] == name]
            for ref in ("classical", "homr"):
                k = f"ref_{ref}"
                pooled = sum(r[k]["step_acc"] * r["n"] for r in group) / sum(r["n"] for r in group)
                vals = [np.median([r[k][v] for r in group if r[k][v] is not None]) for v in ("step_acc", "flat", "tilt_deg")]
                summary += f"| {name} | {ref} | {pooled:.4f} | " + " | ".join(f"{v:.4f}" for v in vals) + " |\n"
        summary += "\nThis repeatedly used CPMS subset is not independent test evidence.\n"
    (args.out / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
