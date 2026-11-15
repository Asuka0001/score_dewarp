"""Frozen-geometry auxiliary training with two-domain validation; no test access.

Cache bottleneck vectors and uniformly sampled decoder pixels, then fit the
existing heads. Full-resolution auxiliary validation runs after selection.
Geometry buffers and parameters remain bit-identical to each anchor.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoredewarp.learn.dataset import build_targets, KINDS
from scoredewarp.learn.infer import load, backward_map
from scoredewarp.learn.selection import geometry_gate, auxiliary_score, DEFAULT_POLICY
from scoredewarp.eval.auxiliary import classification_metrics, segmentation_metrics, heatmap_metrics
from scoredewarp.eval.geometry import geometry_metrics, orientation_error_rate
from scoredewarp.eval.metrics import evaluate
from scoredewarp.core.pipeline import _detect
from train import check_split

ANCHORS = {"original": "outputs/deepscores_finetune/full_ca_lr1e-3_s7/best.pt",
           "adapted": "outputs/structure_ablation_20261115_v2/data_only/best.pt"}
VAL_ROOTS = {"synthetic": "data/prepared/structure_suite_20261115_v2/val",
             "deepscores": "data/prepared/deepscores_full_v3/val"}
TRAIN = "outputs/structure_ablation_20261115_v2/train_replay"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_fingerprint(net):
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        if not name.startswith(("head_kind.", "head_seg.", "head_struct.")):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def summarize_geometry(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["domain"]].append(row["geometry"])
        if row["domain"] == "synthetic":
            groups["synthetic/" + row["kind_name"]].append(row["geometry"])
    return {name: {key: mean([r.get(key) for r in rows]) for key in
                  ("coord_epe_pct", "ad_pct", "local_distortion_pct", "coverage",
                   "orientation_error_rate", "staff_straight", "jp_baseline")}
            for name, rows in groups.items()}


def feature_cache(anchor, path, device):
    marker = path / "complete.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        if info["anchor_sha256"] != sha(ROOT / ANCHORS[anchor]):
            raise ValueError("Cache checkpoint changed")
        return info
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Incomplete cache; use a fresh output directory: {path}")
    path.mkdir(parents=True)
    (path / "dense_val").mkdir()
    net, size = load(ROOT / ANCHORS[anchor], device)
    net.requires_grad_(False).eval()
    train = sorted(p for p in (ROOT / TRAIN).iterdir() if (p / "gt.json").exists())
    for val in VAL_ROOTS.values():
        check_split(ROOT / TRAIN, ROOT / val)
    jobs = [("train", p) for p in train]
    for domain, val in VAL_ROOTS.items():
        jobs += [(domain, p) for p in sorted((ROOT / val).iterdir()) if (p / "gt.json").exists()]
    # Reuse matching-width synthetic detector outputs from the previous stage.
    old_rows = json.loads((ROOT / "outputs/structure_ablation_20261115_v2/comparison_val/per_sample.json").read_text())
    old_name = "full_ca_lr1e-3_s7" if anchor == "original" else "data_only"
    old_config = json.loads((ROOT / "outputs/structure_ablation_20261115_v2/comparison_val/config.json").read_text())
    if old_config["checkpoint"][old_name]["sha256"] != sha(ROOT / ANCHORS[anchor]):
        raise ValueError("Cached detector baseline used a different model")
    old_types = {r["sample"]: r["page_type"] for r in old_rows if r["method"] == old_name}
    records, train_kind, val_kind = [], [], []
    pixels = {split: {k: [] for k in ("x", "seg", "struct")} for split in ("train", "synthetic")}
    start = time.time()
    with torch.no_grad():
        for i, (domain, folder) in enumerate(jobs):
            d = build_targets(str(folder), tuple(size))
            gt = json.loads((folder / "gt.json").read_text())
            image = torch.from_numpy(d["image"])[None].to(device)
            dense, bottleneck = net.forward_features(image)
            vector = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)[0].cpu().numpy()
            key = (domain, folder.name, int(d["kind"]))
            (train_kind if domain == "train" else val_kind).append((*key, vector))
            if d["seg_valid"] and d["struct_valid"] and domain in pixels:
                count = 8192 if domain == "train" else 16384
                seed = int(hashlib.sha256(str(folder.relative_to(ROOT)).encode()).hexdigest()[:16], 16)
                indices = np.random.default_rng(seed).choice(size[0] * size[1], count, replace=False)
                feature = dense[0].cpu().numpy()
                pixels[domain]["x"].append(feature.reshape(feature.shape[0], -1).T[indices].astype(np.float16))
                pixels[domain]["seg"].append(d["seg"].ravel()[indices].astype(np.uint8))
                pixels[domain]["struct"].append(d["struct"].reshape(3, -1).T[indices].astype(np.uint8))
                if domain != "train":
                    np.save(path / "dense_val" / (folder.name + ".npy"), feature.astype(np.float16))
                    np.savez_compressed(path / "dense_val" / (folder.name + ".targets.npz"),
                                        seg=d["seg"].astype(np.uint8), struct=d["struct"].astype(np.uint8),
                                        inst_h=d["inst_h"].astype(np.uint16))
            if domain != "train":
                f = net.head_fwd(dense)
                pred = f[:, :2].sigmoid()[0].cpu().numpy().transpose(1, 2, 0)
                page = f[:, 2:3].sigmoid()[0, 0].cpu().numpy() > .5
                target = d["fwd"].transpose(1, 2, 0)
                valid = d["page"][0] > .5
                aspect = gt["meta"]["size"][1] / gt["meta"]["size"][0]
                geometry = geometry_metrics(target[::8, ::8] * [1, aspect], pred[::8, ::8] * [1, aspect],
                                            valid[::8, ::8], page[::8, ::8])
                geometry["coord_epe_pct"] = float(np.linalg.norm(pred - target, axis=-1)[valid].mean() * 100)
                geometry["coverage"] = float(page[valid].mean())
                geometry["orientation_error_rate"] = orientation_error_rate(target[::8, ::8], pred[::8, ::8], valid[::8, ::8])
                photo = cv2.imread(str(folder / "photo.jpg"))
                def mapper(p):
                    xy = np.asarray(p) / [photo.shape[1], photo.shape[0]] * size
                    return cv2.remap(pred, xy[:, 0].astype(np.float32).reshape(-1, 1),
                                     xy[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE).reshape(-1, 2) * [1240, 1754]
                def point_valid(p):
                    xy = np.asarray(p) / [photo.shape[1], photo.shape[0]] * size
                    return cv2.remap(page.astype(np.uint8), xy[:, 0].astype(np.float32).reshape(-1, 1),
                                     xy[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT).reshape(-1) > 0
                music = evaluate(gt, mapper, point_valid=point_valid, out_size=(1240, 1754))
                for name in ("staff_straight", "jp_baseline"):
                    geometry[name] = music.get(name)
                if domain == "synthetic":
                    manual = old_types[folder.name]
                else:
                    out = {"fwd": f[:, :2].sigmoid().cpu(), "page": f[:, 2:3].cpu()}
                    bm = backward_map(out, (photo.shape[1], photo.shape[0]), (1240, 1754))
                    manual = _detect(bm.apply(photo))[2].page_type
                records.append({"domain": domain, "sample": folder.name, "kind": int(d["kind"]),
                                "kind_name": gt["meta"]["kind"], "manual": KINDS.get(manual, 3),
                                "geometry": geometry})
            if (i + 1) % 25 == 0:
                print(anchor, "features", i + 1, "/", len(jobs), "seconds", round(time.time() - start), flush=True)
    for split, rows in (("train", train_kind), ("val", val_kind)):
        np.savez(path / (split + "_kind.npz"), x=np.stack([r[3] for r in rows]),
                 y=np.array([r[2] for r in rows]), domain=np.array([r[0] for r in rows]),
                 sample=np.array([r[1] for r in rows]))
    for split, arrays in pixels.items():
        for name, values in arrays.items():
            np.save(path / (split + "_" + name + ".npy"), np.concatenate(values))
    info = {"anchor": anchor, "anchor_sha256": sha(ROOT / ANCHORS[anchor]), "size": list(size),
            "geometry_fingerprint": geometry_fingerprint(net), "records": records,
            "geometry": summarize_geometry(records), "train_pages": len(train_kind), "val_pages": len(val_kind),
            "pixel_sampling": {"train_per_page": 8192, "val_per_page": 16384, "uniform_without_replacement": True},
            "features_dtype": "float16 pixels / float32 pooled bottleneck", "elapsed_s": time.time() - start}
    save_json(marker, info)
    return info


@torch.no_grad()
def validate_kind(net, cache, device, threshold=.8):
    data = np.load(cache / "val_kind.npz")
    x = torch.from_numpy(data["x"]).to(device)
    probs = net.head_kind[2:](x).softmax(1).cpu().numpy()
    records = json.loads((cache / "complete.json").read_text())["records"]
    by_key = {(r["domain"], r["sample"]): r for r in records}
    results = {}
    for domain in VAL_ROOTS:
        select = data["domain"] == domain
        truth = data["y"][select]
        manual = np.array([by_key[(domain, s)]["manual"] for s in data["sample"][select]])
        pred, confidence = probs[select].argmax(1), probs[select].max(1)
        fused = np.where(confidence >= threshold, pred, manual)
        results[domain] = {"kind": classification_metrics(truth, pred),
                           "manual": classification_metrics(truth, manual),
                           "fused": classification_metrics(truth, fused),
                           "confidence_coverage": float((confidence >= threshold).mean())}
    return results


@torch.no_grad()
def validate_pixels(net, x, seg, struct):
    confusion = np.zeros((5, 5), np.int64)
    tp, predicted, truth = np.zeros(3, np.int64), np.zeros(3, np.int64), np.zeros(3, np.int64)
    for start in range(0, len(x), 131072):
        features, labels, target = x[start:start+131072], seg[start:start+131072], struct[start:start+131072] > .5
        p = F.linear(features, net.head_seg.weight[:, :, 0, 0], net.head_seg.bias).argmax(1)
        confusion += torch.bincount((labels * 5 + p).cpu(), minlength=25).reshape(5, 5).numpy()
        p = F.linear(features, net.head_struct.weight[:, :, 0, 0], net.head_struct.bias) >= 0
        tp += (p & target).sum(0).cpu().numpy()
        predicted += p.sum(0).cpu().numpy()
        truth += target.sum(0).cpu().numpy()
    return {"seg": segmentation_metrics(confusion), "struct": heatmap_metrics(tp, predicted, truth)}


def checkpoint(net, size, config, metrics, epoch):
    return {"model": net.state_dict(), "arch": net.arch, "size": size, "config": config,
            "metrics": metrics, "epoch": epoch, "auxiliary_heads": {"kind": {"trained": True, "ready": False},
            "seg": {"trained": config["stage"] == "pixels", "ready": False},
            "struct": {"trained": config["stage"] == "pixels", "ready": False}}}


def train_stage(anchor, stage, cache, out, init, device, reference):
    if (out / "report.json").exists():
        return json.loads((out / "report.json").read_text())
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Incomplete stage: {out}")
    out.mkdir(parents=True)
    torch.manual_seed(7)
    net, size = load(init, device)
    net.requires_grad_(False).eval()
    heads = [net.head_kind] if stage == "kind" else [net.head_seg, net.head_struct]
    for head in heads:
        head.requires_grad_(True)
    epochs, batch, lr = (120, 64, .003) if stage == "kind" else (20, 32768, .005)
    config = {"anchor": anchor, "stage": stage, "seed": 7, "epochs": epochs, "batch": batch, "lr": lr,
              "geometry_frozen": True, "geometry_policy": DEFAULT_POLICY,
              "selection": "equal-domain kind balanced accuracy" if stage == "kind" else "0.5 seg mIoU + 0.5 structure macro F1 on fixed sampled validation pixels",
              "threshold": .8, "pixel_threshold": .5, "init_sha256": sha(init)}
    save_json(out / "config.json", config)
    cache_info = json.loads((cache / "complete.json").read_text())
    deployment_gate = geometry_gate(cache_info["geometry"], reference)
    if stage == "kind":
        data = np.load(cache / "train_kind.npz")
        x = torch.from_numpy(data["x"]).to(device)
        y = torch.from_numpy(data["y"]).long().to(device)
        weights = 1 / torch.from_numpy(np.bincount(data["y"], minlength=3)).float().to(device)
        weights = weights / weights.mean()
    else:
        x = torch.from_numpy(np.load(cache / "train_x.npy").astype(np.float32)).to(device)
        y = torch.from_numpy(np.load(cache / "train_seg.npy").astype(np.int64)).to(device)
        target = torch.from_numpy(np.load(cache / "train_struct.npy").astype(np.float32)).to(device)
        vx = torch.from_numpy(np.load(cache / "synthetic_x.npy").astype(np.float32)).to(device)
        vy = torch.from_numpy(np.load(cache / "synthetic_seg.npy").astype(np.int64)).to(device)
        vt = torch.from_numpy(np.load(cache / "synthetic_struct.npy").astype(np.float32)).to(device)
        counts = torch.from_numpy(np.bincount(np.load(cache / "train_seg.npy"), minlength=5)).float().to(device)
        weights = counts.clamp_min(1).rsqrt()
        weights = weights / weights.mean()
    optimizer = torch.optim.AdamW((p for h in heads for p in h.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history, best, start = [], -float("inf"), time.time()
    initial = validate_kind(net, cache, device)
    if stage == "pixels":
        initial["synthetic"].update(validate_pixels(net, vx, vy, vt))
    save_json(out / "initial.json", initial)
    for ep in range(1, epochs + 1):
        order = torch.randperm(len(x), device=device)
        total = 0.
        for indices in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            features = x[indices]
            if stage == "kind":
                loss = F.cross_entropy(net.head_kind[2:](features), y[indices], weight=weights)
            else:
                seg = F.linear(features, net.head_seg.weight[:, :, 0, 0], net.head_seg.bias)
                struct = F.linear(features, net.head_struct.weight[:, :, 0, 0], net.head_struct.bias)
                loss = F.cross_entropy(seg, y[indices], weight=weights) + F.binary_cross_entropy_with_logits(
                    struct, target[indices], pos_weight=torch.tensor(20., device=device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite auxiliary loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        scheduler.step()
        metrics = validate_kind(net, cache, device)
        if stage == "pixels":
            metrics["synthetic"].update(validate_pixels(net, vx, vy, vt))
        score = auxiliary_score(metrics, stage)
        row = {"epoch": ep, "loss": total / len(x), "score": score, "metrics": metrics,
               "elapsed_s": time.time() - start}
        history.append(row)
        if score > best:
            best = score
            torch.save(checkpoint(net, size, config, metrics, ep), out / "best.pt")
        if ep == 1 or ep % (10 if stage == "kind" else 2) == 0:
            print(anchor, stage, ep, "/", epochs, "score", round(score, 4), "seconds", round(time.time()-start), flush=True)
        save_json(out / "history.json", history)
    selected, _ = load(out / "best.pt", device)
    assert geometry_fingerprint(selected) == cache_info["geometry_fingerprint"]
    report = {"best_epoch": max(history, key=lambda r: r["score"])["epoch"], "best_score": best,
              "geometry_unchanged": True, "deployment_geometry_gate": deployment_gate,
              "elapsed_s": time.time() - start, "best_metrics": max(history, key=lambda r: r["score"])["metrics"],
              "checkpoint_sha256": sha(out / "best.pt")}
    save_json(out / "report.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("outputs/auxiliary_20261115"))
    ap.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"status": "running", "completed": []}
    def status(current):
        state["current"] = current
        save_json(args.out / "status.json", state)
    protocol = {"anchors": {name: {"path": path, "sha256": sha(ROOT / path)} for name, path in ANCHORS.items()},
                "training": TRAIN, "validation": VAL_ROOTS, "test_access": False,
                "policy": DEFAULT_POLICY, "kind_confidence": .8,
                "kind_ready": "synthetic fused balanced accuracy >= detector+0.05; DeepScores fused accuracy >= detector-0.01; original-reference geometry gate passed",
                "training_mode": "frozen backbone/geometry and BatchNorm buffers; cached feature head fitting"}
    sources = ["scripts/train_auxiliary.py", "scoredewarp/learn/model.py", "scoredewarp/learn/dataset.py",
               "scoredewarp/learn/selection.py", "scoredewarp/eval/auxiliary.py"]
    protocol["source_sha256"] = {p: sha(ROOT / p) for p in sources}
    protocol["sampling"] = {"train_pixels_per_page": 8192, "val_pixels_per_page": 16384,
                            "full_validation_after_selection": True}
    protocol["kind_training"] = {"epochs": 120, "batch": 64, "lr": .003, "class_balanced_ce": True}
    protocol["pixel_training"] = {"epochs": 20, "batch": 32768, "lr": .005, "struct_pos_weight": 20,
                                  "seg_weights": "inverse square root class frequency"}
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing experiment protocol differs")
    save_json(protocol_path, protocol)
    snapshot = args.out / "source_code"
    snapshot.mkdir(exist_ok=True)
    for path in sources:
        (snapshot / path.replace("/", "__")).write_bytes((ROOT / path).read_bytes())
    try:
        caches = {}
        for anchor in ANCHORS:
            status("cache_" + anchor)
            caches[anchor] = feature_cache(anchor, args.out / ("cache_" + anchor), args.device)
            state["completed"].append(state["current"])
        reference = caches["original"]["geometry"]
        for stage in ("kind", "pixels"):
            for anchor in ANCHORS:
                status(anchor + "_" + stage)
                init = ROOT / ANCHORS[anchor] if stage == "kind" else args.out / (anchor + "_kind/best.pt")
                train_stage(anchor, stage, args.out / ("cache_" + anchor), args.out / (anchor + "_" + stage), init, args.device, reference)
                state["completed"].append(state["current"])
        state["status"] = "trained"
        status(None)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        status(state.get("current"))
        raise


if __name__ == "__main__":
    main()
