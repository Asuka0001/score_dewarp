"""Run the multi-seed replication, full-data training and hybrid validation.

One GPU training job at a time. Never evaluates the test split. Status and logs
survive the terminal session in outputs/next_stage/. Re-running skips completed
training jobs, but refuses to overwrite an interrupted checkpoint.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/next_stage"
PYTHON = str(ROOT / ".venv/bin/python")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = ap.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "runner.lock"
    import fcntl
    lock_file = lock.open("a")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"pid": os.getpid(), "status": "running", "completed": [], "device": args.device}

    def status(**changes):
        state.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
        temp = OUT / "status.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(OUT / "status.json")

    def run(name, command):
        status(current=name, command=command)
        print(name, flush=True)
        with (OUT / (name + ".log")).open("a") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        state["completed"].append(name)
        status()

    def train(name, data, val, seed, pretrained, patience=0):
        dest = Path("outputs/deepscores_finetune") / name
        checkpoint = dest / "best.pt"
        if not (dest / "best.report.json").exists():
            command = [PYTHON, "scripts/train.py", "--data", data, "--val-data", val,
                       "--geometry-only", "--coord-input", "--context", "attn",
                       "--epochs", "60", "--patience", str(patience), "--bs", "8",
                       "--size", "384", "512", "--lr", "0.001", "--seed", str(seed),
                       "--device", args.device, "--out", str(checkpoint)]
            if pretrained:
                command += ["--init", "outputs/doc3d_pretrain/best.pt"]
            run("train_" + name, command)
        run("eval_" + name, [PYTHON, "scripts/evaluate_geometry.py", "--data", val,
                            "--checkpoint", str(checkpoint), "--out", str(dest / "eval_val"), "--resume"])
        return checkpoint

    try:
        status()
        pilot = "data/prepared/deepscores_pilot"
        for seed in (17, 27):
            for arm in ("scratch", "ft"):
                train(f"{arm}_ca_lr1e-3_s{seed}", pilot + "/train", pilot + "/val", seed, arm == "ft")
        pairs = [f"ft_ca_lr1e-3_s{s}:scratch_ca_lr1e-3_s{s}" for s in (7, 17, 27)]
        run("report_multiseed", [PYTHON, "scripts/report_finetune.py", "--pairs", *pairs,
                                 "--out", "reports/deepscores_multiseed.md"])
        full = "data/prepared/deepscores_full_v3"
        checkpoint = train("full_ca_lr1e-3_s7", full + "/train", full + "/val", 7, True, patience=12)
        # Same trained model and exact same sample set for all hybrid variants.
        run("hybrid_full_val", [PYTHON, "scripts/evaluate_geometry.py", "--data", full + "/val",
                                "--checkpoint", str(checkpoint), "--hybrid", "--out",
                                "outputs/hybrid/full_val", "--resume"])
        status(status="complete", current=None)
    except BaseException as exc:
        status(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
