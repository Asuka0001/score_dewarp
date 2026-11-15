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
    net.auxiliary_heads = sd.get("auxiliary_heads", {})
    net.eval().to(device)
    return net, sd.get("size", (384, 512))


def kind_decision(net, out, detector_type="unknown"):
    """Use a validated kind head at fixed confidence, otherwise keep detection."""
    metadata = getattr(net, "auxiliary_heads", {}).get("kind", {})
    logits = out["kind"][0].detach()
    finite = bool(torch.isfinite(logits).all())
    probabilities = logits.softmax(0) if finite else None
    confidence = float(probabilities.max()) if finite else None
    predicted = KIND_NAMES[int(probabilities.argmax())] if finite else "unknown"
    threshold = metadata.get("confidence_threshold", .8)
    valid_threshold = isinstance(threshold, (int, float)) and np.isfinite(threshold) and 0 <= threshold <= 1
    ready = bool(metadata.get("trained") and metadata.get("ready") and metadata.get("deployment_geometry_passed"))
    use = bool(ready and finite and valid_threshold and confidence >= threshold)
    return {"page_type": predicted if use else detector_type,
            "source": "network_kind" if use else "structure_detector",
            "detector_page_type": detector_type, "network_page_type": predicted,
            "confidence": confidence, "threshold": threshold if valid_threshold else None,
            "validated_head": ready}


def apply_kind_decision(result, net, out):
    decision = kind_decision(net, out, result.structures.page_type)
    result.page_type = decision["page_type"]
    result.stages["kind_decision"] = decision
    return result


@torch.no_grad()
def predict(net, img, size=(384, 512), out_size=(1240, 1754), device="cpu"):
    H0, W0 = img.shape[:2]
    x = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(device)
    o = net(x)
    # Geometry-only checkpoints have an untrained kind head; callers should use
    # the structural detector's page_type for their exported results.
    kind = KIND_NAMES[int(o["kind"][0].argmax())]
    return backward_map(o, (W0, H0), out_size), kind, o


def backward_map(out, photo_size, out_size=(1240, 1754)):
    """Invert predicted coordinates using only the predicted page mask.

    A sparse/constant page field is an explicit failure, never a fake identity
    rectification. Shares pixel conventions with eval.geometry.network_points.
    """
    W0, H0 = photo_size
    fwd = out["fwd"][0].detach().cpu().numpy()
    page = out["page"][0, 0].detach().sigmoid().cpu().numpy() > .5
    page &= np.isfinite(fwd).all(0)
    if page.sum() < 6:
        raise ValueError("Not enough predicted page pixels to invert the learned map")
    h, w = page.shape
    ys, xs = np.nonzero(page)
    src = np.stack([xs * W0 / w, ys * H0 / h], 1)
    dst = np.stack([fwd[0, ys, xs] * out_size[0], fwd[1, ys, xs] * out_size[1]], 1)
    if np.linalg.matrix_rank(dst - dst.mean(0)) < 2:
        raise ValueError("Predicted coordinate field is collapsed")
    inv = LinearNDInterpolator(dst, src)
    bm = BackwardMap.from_function(lambda q: inv(q), out_size, stride=8)
    return bm
