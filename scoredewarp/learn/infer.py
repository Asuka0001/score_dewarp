"""Use a trained ScoreDewarpNet: dense forward coords -> BackwardMap -> rectified image."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator

from ..core.transforms import BackwardMap
from .model import build_model

KIND_NAMES = ["staff", "jianpu", "mixed"]


def load(ckpt, device="cpu"):
    sd = torch.load(ckpt, map_location=device)
    net = build_model(sd)
    net.load_state_dict(sd["model"])
    net.eval().to(device)
    return net, sd.get("size", (384, 512))


@torch.no_grad()
def predict(net, img, size=(384, 512), out_size=(1240, 1754), device="cpu"):
    H0, W0 = img.shape[:2]
    x = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(device)
    o = net(x)
    fwd = o["fwd"][0].cpu().numpy()                     # (2,h,w) in [0,1]
    page = torch.sigmoid(o["page"][0, 0]).cpu().numpy() > 0.5
    kind = KIND_NAMES[int(o["kind"][0].argmax())]
    h, w = page.shape
    ys, xs = np.nonzero(page)
    src = np.stack([xs * W0 / w, ys * H0 / h], 1)
    dst = np.stack([fwd[0, ys, xs] * out_size[0], fwd[1, ys, xs] * out_size[1]], 1)
    inv = LinearNDInterpolator(dst, src)
    bm = BackwardMap.from_function(lambda q: inv(q), out_size, stride=8)
    return bm, kind, o
