"""Run comparable geometry-only baselines on a prepared validation split.

Example: python scripts/evaluate_geometry.py --data data/prepared/deepscores_pilot/val --out outputs/baselines/deepscores
Results are saved per sample, so interrupted runs can be resumed with --resume.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.core.pipeline import Config, dewarp
from scoredewarp.eval.geometry import geometry_metrics, sample_geometry, network_points, orientation_error_rate


def classical_sample(folder, width):
    cv2.setNumThreads(1)
    image, meta, points, target, valid = sample_geometry(folder)
    rows = []
    configs = {"input": None, "A": Config(use_surface=False, use_field=False, coarse_width=width),
               "A+B": Config(use_field=False, coarse_width=width), "A+B+C": Config(coarse_width=width)}
    for name, cfg in configs.items():
        start = time.perf_counter()
        row = {"sample": Path(folder).name, "kind": meta["kind"], "method": name}
        try:
            if cfg is None:
                pred, pred_valid = points, None
            else:
                res = dewarp(image, cfg)
                pred = res.forward_points(points.reshape(-1, 2)).reshape(points.shape)
                pred_valid = ((pred >= 0).all(-1) & (pred[..., 0] < res.bmap.out_size[0]) &
                              (pred[..., 1] < res.bmap.out_size[1]))
            row.update(geometry_metrics(target, pred, valid, pred_valid), status="ok")
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}", coverage=0.,
                       ad_pct=None, local_distortion_pct=None)
        row["time_s"] = time.perf_counter() - start
        rows.append(row)
    return rows


def summarize(rows):
    summary = {}
    for name in dict.fromkeys(r["method"] for r in rows):
        group = [r for r in rows if r["method"] == name]
        record = {"samples": len(group), "failed": sum(r["status"] != "ok" for r in group),
                  "undefined_ad": sum(r.get("ad_pct") is None for r in group)}
        metrics = ["coverage", "ad_pct", "local_distortion_pct", "time_s"]
        metrics += [k for k in ("coord_epe_pct_all_gt", "orientation_error_rate") if any(k in r for r in group)]
        for metric in metrics:
            values = [r[metric] for r in group if r.get(metric) is not None and np.isfinite(r[metric])]
            record[metric + "_mean"] = float(np.mean(values)) if values else None
            record[metric + "_median"] = float(np.median(values)) if values else None
        summary[name] = record
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--coarse-width", type=int, default=1600)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry-failed", action="store_true", help="with --resume, retry failed samples and archive their previous rows")
    ap.add_argument("--checkpoint", type=Path, help="evaluate learned field instead of classical baselines")
    args = ap.parse_args()
    samples = sorted(p for p in args.data.iterdir() if (p / "gt.json").exists())
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        ap.error("No samples found")
    args.out.mkdir(parents=True, exist_ok=True)
    config = {"data": str(args.data.resolve()), "samples": [p.name for p in samples],
              "coarse_width": args.coarse_width, "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
              "max_grid_points": 4096, "alignment": "orientation-preserving similarity, prediction to flat page"}
    if args.checkpoint:
        with args.checkpoint.open("rb") as f:
            config["checkpoint_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    config_path = args.out / "config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            ap.error("Output exists; use a fresh output directory, or --resume with identical settings")
    config_path.write_text(json.dumps(config, indent=2))
    row_dir = args.out / "samples"
    row_dir.mkdir(exist_ok=True)
    pending = []
    for p in samples:
        result_path = row_dir / (p.name + ".json")
        if not result_path.exists():
            pending.append(p)
        elif args.retry_failed and args.resume:
            previous = json.loads(result_path.read_text())
            if any(r["status"] != "ok" for r in previous):
                archive = args.out / "failures_before_retry"
                archive.mkdir(exist_ok=True)
                (archive / result_path.name).write_text(json.dumps(previous, indent=2))
                pending.append(p)
    if args.checkpoint:
        import torch
        from scoredewarp.learn.infer import load
        net, size = load(args.checkpoint)
        for i, folder in enumerate(pending):
            image, meta, points, target, valid = sample_geometry(folder)
            x = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)[..., ::-1].copy()
            start = time.perf_counter()
            with torch.no_grad():
                out = net(torch.from_numpy(x.transpose(2, 0, 1)).float()[None] / 255)
            pred, page = network_points(out, points, (image.shape[1], image.shape[0]), meta["size"][1] / meta["size"][0])
            row = {"sample": folder.name, "kind": meta["kind"], "method": "network", "status": "ok",
                   **geometry_metrics(target, pred, valid, page), "time_s": time.perf_counter() - start}
            row["coord_epe_pct_all_gt"] = float(np.linalg.norm((pred - target)[valid], axis=-1).mean() * 100)
            row["orientation_error_rate"] = orientation_error_rate(target, pred, valid)
            (row_dir / (folder.name + ".json")).write_text(json.dumps([row], indent=2))
            if (i + 1) % 10 == 0:
                print(f"network {i+1}/{len(pending)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(classical_sample, str(p), args.coarse_width): p for p in pending}
            for i, job in enumerate(as_completed(jobs)):
                folder = jobs[job]
                rows = job.result()
                (row_dir / (folder.name + ".json")).write_text(json.dumps(rows, indent=2))
                print(f"{i+1}/{len(pending)} {folder.name}", flush=True)
    rows = [r for p in samples for r in json.loads((row_dir / (p.name + ".json")).read_text())]
    summary = summarize(rows)
    (args.out / "per_sample.json").write_text(json.dumps(rows, indent=2))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["| method | samples | failures | undefined AD | coverage mean | AD % median | local distortion % median | seconds mean |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return "—" if v is None else f"{v:.4f}"
    for name, s in summary.items():
        lines.append(f"| {name} | {s['samples']} | {s['failed']} | {s['undefined_ad']} | " +
                     " | ".join(fmt(s[k]) for k in ("coverage_mean", "ad_pct_median", "local_distortion_pct_median", "time_s_mean")) + " |")
    lines += ["", "AD removes global translation/rotation/uniform scale. Local distortion compares adjacent grid edges after the same alignment.",
              "Errors use valid predicted points; always read them together with coverage and failures. Values are per-page aggregates."]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
