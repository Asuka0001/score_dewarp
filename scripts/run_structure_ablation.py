"""Run frozen benchmark -> two controlled fine-tunes -> validation/test checks.

Uses one accelerator job at a time and saves status/logs. Completed stages can
be skipped; incomplete output directories are never silently overwritten.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
BASELINE = Path("outputs/deepscores_finetune/full_ca_lr1e-3_s7/best.pt")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_replay(suite, out):
    """Freeze 100 DeepScores train pages; replay is identical in both arms."""
    manifest = json.loads((suite / "manifest.json").read_text())
    if manifest["counts"] != {"train": 300, "val": 90, "test": 90}:
        raise ValueError("This fixed experiment requires the default 300/90/90 suite")
    for row in manifest["samples"]:
        if not row.get("projection", {}).get("valid"):
            raise ValueError(f"Suite has no valid projection audit: {row['path']}")
        for name, digest in row["sha256"].items():
            if sha(suite / row["path"] / name) != digest:
                raise ValueError(f"Frozen sample was modified: {row['path']}/{name}")
    source = ROOT / "data/prepared/deepscores_full_v3/train"
    candidates = sorted(p for p in source.iterdir() if (p / "gt.json").exists())
    indices = sorted(np.random.default_rng(7).choice(len(candidates), 100, replace=False).tolist())
    samples = [suite / r["path"] for r in manifest["samples"] if r["path"].startswith("train/")]
    samples += [candidates[i] for i in indices]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        sample = sample.resolve()
        dest = out / sample.name
        if dest.is_symlink():
            if dest.resolve() != sample:
                raise ValueError(f"Different replay target: {dest}")
        elif dest.exists():
            raise ValueError(f"Replay path occupied: {dest}")
        else:
            dest.symlink_to(sample, target_is_directory=True)
        rows.append({"name": sample.name, "source": str(sample.relative_to(ROOT)),
                     "gt_sha256": sha(sample / "gt.json"), "photo_sha256": sha(sample / "photo.jpg")})
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError("Replay name collision")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", type=Path, default=Path("data/prepared/structure_suite_20261115_v2"))
    ap.add_argument("--out", type=Path, default=Path("outputs/structure_ablation_20261115_v2"))
    ap.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    args = ap.parse_args()
    args.suite = args.suite.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    train = args.out / "train_replay"
    replay = prepare_replay(args.suite, train)
    sources = ["scripts/train.py", "scripts/prepare_structure_suite.py", "scripts/evaluate_materials.py",
               "scripts/evaluate_geometry.py", "scoredewarp/learn/model.py", "scoredewarp/learn/dataset.py",
               "scoredewarp/learn/infer.py", "scoredewarp/synth/distort.py", "scoredewarp/synth/render.py",
               "scoredewarp/core/pipeline.py", "scoredewarp/core/structures.py", "scoredewarp/core/transforms.py",
               "scoredewarp/eval/geometry.py", "scoredewarp/eval/metrics.py", "scoredewarp/eval/structures.py"]
    protocol = {"suite": str(args.suite.relative_to(ROOT)), "suite_manifest_sha256": sha(args.suite / "manifest.json"),
                "baseline": str(BASELINE), "baseline_sha256": sha(ROOT / BASELINE),
                "train": len(replay), "synthetic_train": 300, "deepscores_replay": 100,
                "val": 90, "test": 90, "seed": 7, "epochs": 24, "batch_size": 8,
                "lr": .0001, "size": [384, 512], "device": args.device,
                "arms": {"data_only": 0., "structure_consistency": 50.},
                "checkpoint_selection": "lowest validation mean per-page UV EPE, no early stopping",
                "frozen_auxiliary_heads": ["seg", "struct", "kind"],
                "evaluation_width": 1240, "test_policy": "evaluate only after both arms finish; no test-based tuning",
                "source_sha256": {p: sha(ROOT / p) for p in sources}, "replay": replay}
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing experiment protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2))
    snapshot = args.out / "source_code"
    snapshot.mkdir(exist_ok=True)
    for path in sources:
        (snapshot / path.replace("/", "__")).write_bytes((ROOT / path).read_bytes())
    state = {"status": "running", "completed": []}

    def status(**changes):
        state.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
        tmp = args.out / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(args.out / "status.json")

    def run(name, command, result):
        marker = args.out / (name + ".complete.json")
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous["command"] != command or not result.exists():
                raise ValueError(f"Cannot reuse {name}")
            state["completed"].append(name)
            return
        status(current=name, command=command)
        print(name, flush=True)
        with (args.out / (name + ".log")).open("a") as log:
            subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if not result.exists():
            raise ValueError(f"Missing result: {result}")
        marker.write_text(json.dumps({"command": command, "result": str(result), "result_sha256": sha(result)}, indent=2))
        state["completed"].append(name)
        status()

    def evaluate(name, split, checkpoints):
        dest = args.out / name
        command = [PYTHON, "scripts/evaluate_materials.py", "--data", str(args.suite / split),
                   "--out", str(dest), "--width", "1240", "--save-images"]
        for ckpt in checkpoints:
            command += ["--checkpoint", str(ckpt)]
        run(name, command, dest / "summary.json")

    try:
        status()
        baseline = ROOT / BASELINE
        evaluate("baseline_val", "val", [baseline])
        checkpoints = [baseline]
        for name, weight in protocol["arms"].items():
            ckpt = args.out / name / "best.pt"
            command = [PYTHON, "scripts/train.py", "--data", str(train), "--val-data", str(args.suite / "val"),
                       "--init", str(baseline), "--out", str(ckpt), "--geometry-only",
                       "--consistency-weight", str(weight), "--coord-input", "--context", "attn",
                       "--epochs", "24", "--patience", "0", "--min-delta", "0.001", "--bs", "8",
                       "--size", "384", "512", "--lr", "0.0001", "--seed", "7", "--device", args.device]
            run("train_" + name, command, ckpt.with_suffix(".report.json"))
            checkpoints.append(ckpt)
        evaluate("comparison_val", "val", checkpoints)
        # All training settings and both best checkpoints are now fixed.
        evaluate("comparison_test", "test", checkpoints)
        for ckpt in checkpoints:
            dest = args.out / "retention_val" / ckpt.parent.name
            command = [PYTHON, "scripts/evaluate_geometry.py", "--data", "data/prepared/deepscores_full_v3/val",
                       "--checkpoint", str(ckpt), "--out", str(dest)]
            run("retention_" + ckpt.parent.name, command, dest / "summary.json")
        run("report", [PYTHON, "scripts/report_structure_ablation.py", "--experiment", str(args.out),
                       "--out", "reports/structure_ablation_20261115.md"], ROOT / "reports/structure_ablation_20261115.md")
        status(status="complete", current=None)
    except BaseException as exc:
        status(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
