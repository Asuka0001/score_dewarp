"""Full-pixel development validation and guarded packaging of auxiliary heads.

No test split is opened. Kind may be enabled only after the predeclared
two-domain quality/geometry gates. Seg/struct remain standalone predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoredewarp.learn.infer import load
from scoredewarp.learn.selection import geometry_gate
from scoredewarp.eval.auxiliary import heatmap_metrics, segmentation_metrics
from train_auxiliary import ANCHORS, sha, save_json, geometry_fingerprint, validate_kind


def average_precision(scores, labels):
    labels = np.asarray(labels, bool)
    if not labels.any():
        return None
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    hit = y.cumsum()[last]
    recall = hit / labels.sum()
    precision = hit / (last + 1)
    return float((np.diff(np.r_[0., recall]) * precision).sum())


@torch.no_grad()
def full_pixel_metrics(net, cache, device):
    groups = {}
    records, ap_scores, ap_truth = [], [], []
    rows = json.loads((cache / "complete.json").read_text())["records"]
    for row in (r for r in rows if r["domain"] == "synthetic"):
        kind, sample = row["kind_name"], row["sample"]
        feature = np.load(cache / "dense_val" / (sample + ".npy")).astype(np.float32)
        if net.auxiliary_context:
            photo = cv2.imread(str(ROOT / "data/prepared/structure_suite_20261115_v2/val" / sample / "photo.jpg"))
            rgb = cv2.resize(photo, (feature.shape[2], feature.shape[1]), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32)/255
            feature = np.concatenate([feature, rgb.transpose(2,0,1)])
        targets = np.load(cache / "dense_val" / (sample + ".targets.npz"))
        x = torch.from_numpy(feature)[None].to(device)
        seg = net.head_seg(x)[0].argmax(0).cpu().numpy()
        probs = net.head_struct(x)[0].sigmoid().cpu().numpy()
        pred = probs >= .5
        truth = targets["struct"] > 0
        relative = f"data/prepared/structure_suite_20261115_v2/val/{sample}"
        seed = int(hashlib.sha256(relative.encode()).hexdigest()[:16],16)
        indices = np.random.default_rng(seed).choice(truth.shape[1]*truth.shape[2],16384,replace=False)
        ap_scores.append(probs.reshape(3,-1).T[indices])
        ap_truth.append(truth.reshape(3,-1).T[indices])
        confusion = np.bincount((targets["seg"].astype(np.int64) * 5 + seg).ravel(), minlength=25).reshape(5,5)
        tp, pp, tt = (pred & truth).sum((1,2)), pred.sum((1,2)), truth.sum((1,2))
        near_gt = np.stack([cv2.dilate(t.astype(np.uint8), np.ones((3,3),np.uint8)) > 0 for t in truth])
        near_pred = np.stack([cv2.dilate(p.astype(np.uint8), np.ones((3,3),np.uint8)) > 0 for p in pred])
        tp_precision, tp_recall = (pred & near_gt).sum((1,2)), (truth & near_pred).sum((1,2))
        line_hit, line_count, coverage = np.zeros(2, int), np.zeros(2,int), [[],[]]
        for channel in (0,1):
            ids = np.unique(targets["inst_h"][truth[channel]])
            for identifier in ids[ids > 0]:
                mask = (targets["inst_h"] == identifier) & truth[channel]
                if mask.sum() < 8:
                    continue
                cov = float(near_pred[channel][mask].mean())
                coverage[channel].append(cov)
                line_count[channel] += 1
                line_hit[channel] += cov >= .5
        for name in ("all",kind):
            if name not in groups:
                groups[name] = {"confusion":np.zeros((5,5),np.int64), "tp":np.zeros(3,np.int64),
                                "pp":np.zeros(3,np.int64),"tt":np.zeros(3,np.int64),
                                "tp_precision":np.zeros(3,np.int64),"tp_recall":np.zeros(3,np.int64),
                                "line_hit":np.zeros(2,np.int64),"line_count":np.zeros(2,np.int64),"pages":0}
            g=groups[name]
            for key,value in (("confusion",confusion),("tp",tp),("pp",pp),("tt",tt),
                              ("tp_precision",tp_precision),("tp_recall",tp_recall),
                              ("line_hit",line_hit),("line_count",line_count)):
                g[key] += value
            g["pages"] += 1
        records.append({"sample":sample,"kind":kind,
                        "line_coverage": [float(np.mean(v)) if v else None for v in coverage]})
    result={}
    for name,g in groups.items():
        tolerant=[]
        for hitp,hitr,p,t in zip(g["tp_precision"],g["tp_recall"],g["pp"],g["tt"]):
            precision,recall = float(hitp/p) if p else 0.,float(hitr/t) if t else 0.
            tolerant.append({"precision":precision if p else None,"recall":recall if t else None,
                             "f1":2*precision*recall/(precision+recall) if precision+recall else (0. if p+t else None)})
        result[name]={"pages":g["pages"], "seg":segmentation_metrics(g["confusion"]),
                      "struct":heatmap_metrics(g["tp"],g["pp"],g["tt"]),"tolerant_1px":tolerant,
                      "line_hit_recall":[float(h/n) if n else None for h,n in zip(g["line_hit"],g["line_count"])],
                      "line_hit_count":g["line_hit"].tolist(),"line_gt_count":g["line_count"].tolist()}
    # Confidence-independent AP on the fixed uniform validation pixel sample.
    scores, labels = np.concatenate(ap_scores), np.concatenate(ap_truth)
    result["all"]["sampled_average_precision"]=[average_precision(scores[:,i],labels[:,i]) for i in range(3)]
    return {"summary":result,"per_sample":records}


def kind_readiness(metrics, gate):
    reasons=[]
    syn, ds = metrics["synthetic"], metrics["deepscores"]
    if syn["fused"]["balanced_accuracy"] < syn["manual"]["balanced_accuracy"]+.05:
        reasons.append("synthetic fused balanced accuracy did not improve by 5 percentage points")
    if ds["fused"]["accuracy"] < ds["manual"]["accuracy"]-.01:
        reasons.append("DeepScores fused accuracy dropped by more than 1 percentage point")
    return {"quality_passed":not reasons,"quality_reasons":reasons,
            "geometry_passed":gate["passed"],"ready":not reasons and gate["passed"]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment",type=Path,default=Path("outputs/auxiliary_20261115"))
    ap.add_argument("--device",default="mps",choices=("cpu","mps","cuda"))
    args=ap.parse_args()
    exp=args.experiment.resolve()
    if json.loads((exp/"status.json").read_text())["status"] not in ("trained","complete"):
        ap.error("Finish auxiliary training first")
    if (exp/"original_spatial/protocol.json").exists() and not (exp/"original_spatial/report.json").exists():
        ap.error("Finish the spatial auxiliary follow-up first")
    reference=json.loads((exp/"cache_original/complete.json").read_text())["geometry"]
    bundle={"test_access":False,"thresholds":{"kind_confidence":.8,"struct_probability":.5,
            "tolerance_pixels":1,"instance_hit_fraction":.5,"instance_min_pixels":8},"anchors":{}}
    for anchor,path in ANCHORS.items():
        cache=exp/("cache_"+anchor)
        info=json.loads((cache/"complete.json").read_text())
        gate=geometry_gate(info["geometry"],reference)
        models={"untrained_heads":ROOT/path,"kind":exp/(anchor+"_kind/best.pt"),
                "pixels":exp/(anchor+"_pixels/best.pt")}
        if anchor=="original" and (exp/"original_spatial/report.json").exists():
            models={"untrained_heads":ROOT/path,"kind":exp/(anchor+"_kind/best.pt"),
                    "linear_pixels":exp/(anchor+"_pixels/best.pt"),"pixels":exp/"original_spatial/best.pt"}
        record={"geometry":info["geometry"],"deployment_geometry_gate":gate,"models":{}}
        baseline_kind=None
        for stage,checkpoint_path in models.items():
            net,size=load(checkpoint_path,args.device)
            assert geometry_fingerprint(net)==info["geometry_fingerprint"]
            kind=validate_kind(net,cache,args.device)
            pixel=full_pixel_metrics(net,cache,args.device) if stage!="kind" else None
            entry={"kind":kind,"pixels":pixel,"checkpoint_sha256":sha(checkpoint_path),"geometry_unchanged":True}
            if stage=="kind":
                baseline_kind={k:v.detach().cpu().clone() for k,v in net.state_dict().items() if k.startswith("head_kind.")}
            if stage=="pixels":
                assert baseline_kind and all(torch.equal(v,net.state_dict()[k].detach().cpu()) for k,v in baseline_kind.items())
            if stage!="untrained_heads":
                ready=kind_readiness(kind,gate)
                entry["kind_readiness"]=ready
                data=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
                data["auxiliary_heads"]={"kind":{"trained":True,"ready":ready["ready"],
                    "confidence_threshold":.8,"validation":"synthetic val 90 + DeepScores val 172; no independent test",
                    "quality_passed":ready["quality_passed"],"deployment_geometry_passed":gate["passed"]},
                    "seg":{"trained":stage in ("pixels","linear_pixels"),"evaluated":stage in ("pixels","linear_pixels"),"ready":False},
                    "struct":{"trained":stage in ("pixels","linear_pixels"),"evaluated":stage in ("pixels","linear_pixels"),"ready":False,
                              "reason":"pixel/line-hit validation only; instance decoding and geometry confidence not calibrated"}}
                destination=checkpoint_path.with_name("validated.pt")
                torch.save(data,destination)
                entry["validated_checkpoint"]=str(destination.relative_to(ROOT))
                entry["validated_sha256"]=sha(destination)
            record["models"][stage]=entry
            print(anchor,stage,"fused synthetic/DS",[round(kind[d]["fused"]["accuracy"],4) for d in ("synthetic","deepscores")],flush=True)
        bundle["anchors"][anchor]=record
    save_json(exp/"evaluation.json",bundle)
    state=json.loads((exp/"status.json").read_text())
    state.update(status="complete",evaluation_complete=True,current=None)
    save_json(exp/"status.json",state)
    print(exp/"evaluation.json",flush=True)


if __name__=="__main__":
    main()
