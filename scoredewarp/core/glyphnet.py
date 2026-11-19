"""Small CNN for jianpu glyph blobs: digit 0-7 or other (``scripts/train_glyph.py``).

The template matcher (``structures.classify_digit``) misses ~27 % of the
digits on the dev jianpu pages (broken strokes, attached ink, unusual fonts).
This 30 k-parameter network classifies one binarised connected component,
the crop the detector makes. Optional (``MetricConfig.glyph="cnn"``); weights
in ``models/glyph/glyph.onnx`` (trained on synthetic glyphs from system fonts
the bench renderer does not use).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

SIZE = 32
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "models" / "glyph" / "glyph.onnx"
_SESSION = None


def prepare(crop: np.ndarray) -> np.ndarray:
    """Binary crop (h, w) -> (SIZE, SIZE) float32, aspect kept, centred, max side 28."""
    h, w = crop.shape
    f = 28.0 / max(h, w)
    nh, nw = max(1, int(round(h * f))), max(1, int(round(w * f)))
    r = cv2.resize(crop.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((SIZE, SIZE), np.float32)
    y0, x0 = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def build_torch():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.MaxPool2d(2))
    return nn.Sequential(block(1, 16), block(16, 32), block(32, 48), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.2), nn.Linear(48, 9))


def available() -> bool:
    return Path(os.environ.get("SCOREDEWARP_GLYPH", DEFAULT_PATH)).exists()


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 1
        _SESSION = ort.InferenceSession(str(os.environ.get("SCOREDEWARP_GLYPH", DEFAULT_PATH)), sess_options=so,
                                        providers=["CPUExecutionProvider"])
    return _SESSION


def classify(crops, thr=0.8):
    """List of binary crops -> list of (digit str or None, probability)."""
    if not crops:
        return []
    x = np.stack([prepare(c) for c in crops])[:, None]
    s = _session()
    logits = s.run(None, {s.get_inputs()[0].name: x})[0]
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    k = p.argmax(1)
    return [(str(int(c)) if c < 8 and q >= thr else None, float(q)) for c, q in zip(k, p.max(1))]
