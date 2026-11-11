"""Training targets for the learned model, built from make_synth.py folders.

Per sample (all aligned with the *input photo*, resized to `size`):
  image      3xHxW float, [0,1]
  fwd        2xHxW  photo pixel -> flat page coords, normalised to [0,1]   (dense flow)
  page       1xHxW  page mask (valid fwd)
  seg        HxW    0 off-page, 1 paper, 2 staff region, 3 jianpu region, 4 text/lyrics
  struct     3xHxW  heatmaps: staff lines, jianpu baselines (digit bottoms), vertical lines
  inst_h     HxW    instance id of horizontal structures (staff line / jianpu baseline), 0 = none
  inst_v     HxW    instance id of vertical structures (barlines / stems), 0 = none
  kind       int    0 staff, 1 jianpu, 2 mixed
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

KINDS = {"staff": 0, "jianpu": 1, "mixed": 2}


def build_targets(folder, size=(384, 512)):
    W, H = size
    img = cv2.imread(os.path.join(folder, "photo.jpg"))
    H0, W0 = img.shape[:2]
    sx, sy = W / W0, H / H0
    with open(os.path.join(folder, "gt.json")) as f:
        gt = json.load(f)
    Wf, Hf = gt["meta"]["size"]
    bm = np.load(os.path.join(folder, "gt_bmap_s2.npy")).astype(np.float32)
    bm = cv2.resize(np.nan_to_num(bm, nan=-1), (W, H), interpolation=cv2.INTER_NEAREST)
    page = (np.isfinite(bm).all(axis=-1) & (bm >= 0).all(axis=-1) &
            (bm[..., 0] <= Wf) & (bm[..., 1] <= Hf)).astype(np.float32)
    fwd = np.stack([bm[..., 0] / Wf, bm[..., 1] / Hf], 0)
    fwd[:, page == 0] = 0

    seg = page.astype(np.int32)
    struct = np.zeros((3, H, W), np.float32)
    inst_h = np.zeros((H, W), np.int32)
    inst_v = np.zeros((H, W), np.int32)
    S = np.array([sx, sy])
    for a in gt["photo_ann"]:
        if a["kind"] == "region":
            poly = (np.array(a["pts"]) * S).astype(np.int32)
            cls = {"staff": 2, "jianpu": 3}.get(a["type"], 4)
            cv2.fillPoly(seg, [poly], cls)
    for a in gt["photo_ann"]:
        if a["kind"] in ("lyric_char", "text_char"):
            p = np.array(a["pts"][:4]) * S
            cv2.fillPoly(seg, [p.astype(np.int32)], 4)
    hid, vid = 1, 1
    rows = {}
    for a in gt["photo_ann"]:
        k = a["kind"]
        p = np.array(a["pts"]) * S
        if k == "staff_line":
            cv2.polylines(struct[0], [p.astype(np.int32)], False, 1.0, 1)
            cv2.polylines(inst_h, [p.astype(np.int32)], False, hid, 1)
            hid += 1
        elif k in ("barline", "stem"):
            cv2.line(struct[2], tuple(p[0].astype(int)), tuple(p[1].astype(int)), 1.0, 1)
            cv2.line(inst_v, tuple(p[0].astype(int)), tuple(p[1].astype(int)), vid, 1)
            vid += 1
        elif k == "jp_char":
            rows.setdefault(a["row"], []).append(p[5])
    for r, pts in rows.items():
        pts = np.array(sorted(pts, key=lambda q: q[0]))
        cv2.polylines(struct[1], [pts.astype(np.int32)], False, 1.0, 1)
        cv2.polylines(inst_h, [pts.astype(np.int32)], False, hid, 1)
        hid += 1
    image = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
    supervision = gt["meta"].get("supervision", {})
    kind = gt["meta"]["kind"]
    if kind not in KINDS and supervision.get("kind", True):
        raise ValueError(f"Unknown kind {kind!r} requires supervision.kind=false")
    return {
        "image": np.ascontiguousarray(image.transpose(2, 0, 1)),
        "fwd": fwd.astype(np.float32),
        "page": page[None],
        "seg": seg.astype(np.int64),
        "struct": struct,
        "inst_h": inst_h.astype(np.int64),
        "inst_v": inst_v.astype(np.int64),
        "kind": KINDS.get(kind, 0),  # placeholder ignored when kind_valid=0
        **{f"{head}_valid": np.float32(supervision.get(head, True))
           for head in ("seg", "struct", "kind")},
    }


try:
    import torch
    from torch.utils.data import Dataset

    class SynthScoreDataset(Dataset):
        def __init__(self, root, size=(384, 512), cache=False):
            self.dirs = sorted(os.path.join(root, d) for d in os.listdir(root)
                               if os.path.isfile(os.path.join(root, d, "gt.json")))
            self.size = size
            self.cache = {} if cache else None

        def __len__(self):
            return len(self.dirs)

        def __getitem__(self, i):
            if self.cache is not None and i in self.cache:
                d = self.cache[i]
            else:
                d = build_targets(self.dirs[i], self.size)
                if self.cache is not None:
                    self.cache[i] = d
            return {k: torch.as_tensor(v) for k, v in d.items()}
except ImportError:  # torch optional for the classical pipeline
    pass
