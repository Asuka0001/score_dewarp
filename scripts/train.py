"""Train with held-out validation, reproducible seeds, best/last checkpoints and curves."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.learn.dataset import SynthScoreDataset
from scoredewarp.learn.model import ScoreDewarpNet, load_init_weights, losses


def geometry_batch(batch):
    batch = dict(batch)
    for head in ("seg", "struct", "kind"):
        batch[head + "_valid"] = torch.zeros_like(batch[head + "_valid"])
    batch["inst_h"] = torch.zeros_like(batch["inst_h"])
    batch["inst_v"] = torch.zeros_like(batch["inst_v"])
    return batch


@torch.no_grad()
def validate(net, loader, device, geometry_only):
    net.eval()
    records = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if geometry_only:
            batch = geometry_batch(batch)
        out = net(batch["image"])
        loss, parts = losses(out, batch)
        page = batch["page"][:, 0] > .5
        # All GT page pixels, regardless of the network's page confidence.
        error = torch.linalg.vector_norm(out["fwd"] - batch["fwd"], dim=1)
        epe = (error * page).flatten(1).sum(1) / page.flatten(1).sum(1).clamp_min(1) * 100
        predicted = out["page"][:, 0].sigmoid() > .5
        intersection = (predicted & page).flatten(1).sum(1)
        union = (predicted | page).flatten(1).sum(1).clamp_min(1)
        iou = intersection / union
        for j in range(len(epe)):
            records.append({"coord_epe_pct": epe[j].item(), "page_iou": iou[j].item()})
    return {k: float(np.mean([r[k] for r in records])) for k in records[0]}


def check_split(train, val):
    if Path(train).resolve() == Path(val).resolve():
        raise ValueError("Training and validation directories must differ")
    def groups(root):
        out = set()
        for path in Path(root).glob("*/gt.json"):
            meta = json.loads(path.read_text())["meta"]
            if "source_group" in meta:
                out.add((meta.get("source"), meta["source_group"]))
        return out
    if groups(train) & groups(val):
        raise ValueError("Source-group overlap between training and validation")


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, key, title in zip(axes, ("train_loss", "val_coord_epe_pct", "val_page_iou"),
                              ("Training loss", "Validation UV EPE (%)", "Validation page IoU")):
        ax.plot(epochs, [r.get(key, float("nan")) for r in history], marker=".")
        ax.set(xlabel="Epoch", title=title)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--val-data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=0, help="stop after N steps (smoke test)")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, nargs=2, default=[384, 512])
    ap.add_argument("--out", default="outputs/ckpt.pt", help="best validation model; last is saved as *.last.pt")
    ap.add_argument("--init", help="initialize weights for fine-tuning")
    ap.add_argument("--geometry-only", action="store_true")
    ap.add_argument("--context", choices=["none", "attn"], default="none",
                    help="global self-attention at the bottleneck (needed for large inputs, see model.py)")
    ap.add_argument("--context-layers", type=int, default=2)
    ap.add_argument("--coord-input", action="store_true", help="append normalised (x, y) input channels (CoordConv)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=.02, help="minimum improvement in UV EPE percentage points")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--cache", action="store_true", help="cache resized targets in RAM")
    args = ap.parse_args()
    if args.epochs < 1 or args.bs < 1:
        ap.error("epochs and bs must be positive")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        ap.error("Output checkpoint exists; select a fresh path")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ds = SynthScoreDataset(args.data, tuple(args.size), cache=args.cache)
    if not len(ds):
        ap.error(f"No samples in {args.data}")
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=False)
    vl = None
    if args.val_data:
        check_split(args.data, args.val_data)
        vs = SynthScoreDataset(args.val_data, tuple(args.size), cache=args.cache)
        if not len(vs):
            ap.error("Validation dataset is empty")
        vl = DataLoader(vs, batch_size=args.bs, num_workers=0)
    net = ScoreDewarpNet(context=args.context, context_layers=args.context_layers,
                         coord_input=args.coord_input).to(dev)
    init_missing = []
    if args.init:
        init_missing = load_init_weights(net, torch.load(args.init, map_location=dev, weights_only=True)["model"])
        if init_missing:
            print(f"init: {len(init_missing)} new context tensors not in {args.init}; started as identity", flush=True)
    if args.geometry_only:
        for head in (net.head_seg, net.head_struct, net.head_kind):
            head.requires_grad_(False)
    opt = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * len(dl)))
    config = {**vars(args), "device_used": dev, "train_samples": len(ds), "val_samples": len(vs) if vl else 0,
              "arch": net.arch, "parameters": sum(p.numel() for p in net.parameters()),
              "init_missing_tensors": len(init_missing),
              "selection_metric": "validation mean per-page UV EPE (%) on all GT page pixels",
              "torch_version": str(torch.__version__)}
    out_path.with_suffix(".config.json").write_text(json.dumps(config, indent=2))
    history, best, best_epoch, wait, step = [], float("inf"), 0, 0, 0
    t0 = time.time()
    initial = validate(net, vl, dev, args.geometry_only) if vl else {}
    history.append({"epoch": 0, "step": 0, "elapsed_s": time.time()-t0, **{"val_"+k:v for k,v in initial.items()}})
    print("device", dev, "train", len(ds), "val", len(vs) if vl else 0, "initial", initial, flush=True)
    stop_reason = "epochs_completed"
    for ep in range(1, args.epochs + 1):
        net.train()
        train_sum, count = 0., 0
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            if args.geometry_only:
                batch = geometry_batch(batch)
            opt.zero_grad(set_to_none=True)
            loss, parts = losses(net(batch["image"]), batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            opt.step()
            sched.step()
            step += 1
            n = batch["image"].shape[0]
            train_sum += loss.item() * n
            count += n
            if step % 100 == 0 or step == 1:
                print(f"epoch {ep} step {step} loss {loss.item():.5f} elapsed {time.time()-t0:.0f}s", flush=True)
            if args.steps and step >= args.steps:
                stop_reason = "step_limit"
                break
        metrics = validate(net, vl, dev, args.geometry_only) if vl else {}
        row = {"epoch": ep, "step": step, "train_loss": train_sum / count,
               "elapsed_s": time.time()-t0, "lr": opt.param_groups[0]["lr"], **{"val_"+k:v for k,v in metrics.items()}}
        history.append(row)
        score = metrics.get("coord_epe_pct", train_sum/count)
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation score")
        improved = score < best
        significant = score < best - args.min_delta
        wait = 0 if significant else wait + 1
        if improved:
            best, best_epoch = score, ep
        checkpoint = {"model": net.state_dict(), "arch": net.arch, "size": tuple(args.size), "epoch": ep, "step": step,
                      "metrics": metrics, "config": config, "best_epoch": best_epoch, "best_score": best,
                      "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "rng_state": torch.get_rng_state()}
        torch.save(checkpoint, out_path.with_suffix(".last.pt"))
        if improved or not vl:
            torch.save(checkpoint, out_path)
        out_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
        plot_history(history, out_path.with_suffix(".curves.png"))
        print(json.dumps(row), "best_epoch", best_epoch, flush=True)
        if args.steps and step >= args.steps:
            break
        if vl and args.patience > 0 and wait >= args.patience:
            stop_reason = "early_stopping"
            break
    report = {"stop_reason": stop_reason, "epochs_run": ep, "steps": step, "elapsed_s": time.time()-t0,
              "best_epoch": best_epoch, "best_score": best, "initial_validation": initial,
              "best_validation": next({k[4:]:v for k,v in r.items() if k.startswith("val_")} for r in history if r["epoch"] == best_epoch),
              "best_checkpoint": str(out_path), "last_checkpoint": str(out_path.with_suffix(".last.pt"))}
    out_path.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
