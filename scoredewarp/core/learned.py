"""Optional learned staff-notation segmentation (detection only, no warping).

The geometric pipeline stays the solver; a small off-the-shelf network only
replaces the hand-made pixel evidence for staff lines and stems, the two cues
where morphology is weakest (thin grey lines under shading, stems pulled ~1 deg
by their noteheads / flags).

Model: homr ``segnet`` (UNet adapted from oemer, trained on DeepScoresV2),
https://github.com/liebharc/homr  -- code and weights AGPL-3.0, training data
CC BY 4.0. The weights are NOT shipped with this repository; download them with
``scripts/download_models.py`` (or point ``SCOREDEWARP_SEGNET`` at the .onnx).
Inference here is our own sliding-window code that mirrors homr's
preprocessing: width set from the staff spacing (``model_width``), CLAHE grey, 320 px windows, half-window stride.

``segment(img)`` returns a uint8 (H, W, 5) map at the photo resolution with the
per-class vote share * 255 for CHANNELS. Warp it with the same BackwardMap as
the image (linear interpolation) and threshold at 127.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

CHANNELS = ("staff", "stem", "notehead", "clef", "symbol")
# homr class index -> CHANNELS index (0 is background)
_HOMR_CLASSES = {4: 0, 1: 1, 2: 2, 3: 3, 5: 4}
SEGNET_NAME = "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f"
SEGNET_URL = f"https://github.com/liebharc/homr/releases/download/onnx_checkpoints/{SEGNET_NAME}.zip"
_REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = _REPO / "data" / "external" / "models" / "homr" / f"{SEGNET_NAME}.onnx"

_SESSION = None


def model_path() -> Path:
    return Path(os.environ.get("SCOREDEWARP_SEGNET", DEFAULT_PATH))


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return model_path().exists()


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        p = model_path()
        if not p.exists():
            raise FileNotFoundError(f"segmentation model not found: {p} (run scripts/download_models.py)")
        prov = ["CPUExecutionProvider"]
        if os.environ.get("SCOREDEWARP_SEG_COREML", "1") == "1" and "CoreMLExecutionProvider" in ort.get_available_providers():
            prov = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _SESSION = ort.InferenceSession(str(p), sess_options=so, providers=prov)
    return _SESSION


def _windows(n, win, step):
    return sorted({max(min(v, n - win), 0) for v in range(0, max(n, win), step)})


def _segment_small(grey: np.ndarray, win=320, batch=8) -> np.ndarray:
    """grey (h, w) uint8 at model scale -> (h, w, 5) vote share in [0, 1]."""
    s = _session()
    h, w = grey.shape
    H, W = max(h, win), max(w, win)
    canvas = np.full((H, W), 255, np.uint8)
    canvas[:h, :w] = grey
    step = win // 2
    pos = [(y, x) for y in _windows(H, win, step) for x in _windows(W, win, step)]
    votes = np.zeros((H, W, len(CHANNELS)), np.float32)
    count = np.zeros((H, W), np.float32)
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        b = np.stack([canvas[y:y + win, x:x + win] for y, x in chunk]).astype(np.float32)
        b = np.repeat(b[:, None], 3, axis=1)                      # N, 3, win, win (BGR of a grey image)
        lab = s.run(None, {"input": b})[0].argmax(1)              # N, win, win
        for (y, x), l in zip(chunk, lab):
            for k, c in _HOMR_CLASSES.items():
                votes[y:y + win, x:x + win, c] += l == k
            count[y:y + win, x:x + win] += 1
    return votes[:h, :w] / np.maximum(count, 1)[:h, :w, None]


def model_width(img: np.ndarray, target_spacing=16.0, lo=1280, hi=4096) -> int:
    """Input width for the network: staff spacing ~``target_spacing`` px.

    homr resizes every page to width 1920, i.e. assumes a full page with
    spacing ~15-20 px there. Low-resolution / small-print photos (Doc3D bench:
    spacing 7 px at 1400 px width) lose whole staves at 1920 (dev 0031: 4 of 7
    staves; at spacing ~13-18 px all found). Spacing from run lengths of the
    binarized photo; pages without a periodic line pattern keep 1920."""
    from .preprocess import binarize, normalize_illumination, to_gray
    from .structures import run_metrics
    W0 = img.shape[1]
    _, sp = run_metrics(binarize(normalize_illumination(to_gray(img))))
    if not sp:
        return 1920
    return int(np.clip(round(W0 * target_spacing / sp), lo, hi))


def segment(img: np.ndarray, *, width=None, cache_dir: str | os.PathLike | None = None) -> np.ndarray:
    """BGR / grey photo -> uint8 (H, W, 5) class vote share * 255 at photo size.

    ``width``: network input width (default ``model_width``)."""
    H0, W0 = img.shape[:2]
    width = int(width or model_width(img))
    cache_dir = cache_dir or os.environ.get("SCOREDEWARP_SEG_CACHE")
    key = None
    if cache_dir:
        key = hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()[:20] + f"_{SEGNET_NAME[:10]}_{width}"
        f = Path(cache_dir) / f"{key}.npz"
        if f.exists():
            small = np.load(f)["seg"]
            return cv2.resize(small, (W0, H0), interpolation=cv2.INTER_LINEAR)
    grey = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = round(H0 * width / W0)
    grey = cv2.resize(grey, (width, h), interpolation=cv2.INTER_AREA if width < W0 else cv2.INTER_LINEAR)
    grey = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8)).apply(grey)
    small = np.round(_segment_small(grey) * 255).astype(np.uint8)
    if key:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(cache_dir) / f"{key}.npz", seg=small)
    return cv2.resize(small, (W0, H0), interpolation=cv2.INTER_LINEAR)


def masks(seg: np.ndarray | None, shape=None, parts=None) -> dict | None:
    """Warped seg map -> {channel: uint8 0/1 mask}.

    ``parts``: channels handed to the detector (default env
    ``SCOREDEWARP_SEG_PARTS``, e.g. "staff" or "stem"; all if unset) -- an
    ablation switch: a missing channel means the classical evidence is used."""
    if seg is None:
        return None
    parts = parts or os.environ.get("SCOREDEWARP_SEG_PARTS")
    if shape is not None and seg.shape[:2] != tuple(shape):
        seg = cv2.resize(seg, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    keep = set(parts.split("+")) if parts else set(CHANNELS)
    return {c: (seg[..., i] > 127).astype(np.uint8) for i, c in enumerate(CHANNELS) if c in keep}
