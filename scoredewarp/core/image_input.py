"""Read common photo formats into the BGR array used by the pipeline."""
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def _open_pillow(path):
    with Image.open(path) as opened:
        opened.load()  # catch truncated conversions before leaving the temp directory
        image = ImageOps.exif_transpose(opened)
        rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_photo(path):
    """Return an oriented uint8 BGR image; originals are never changed.

    Pillow reads the ordinary formats. On macOS ImageIO, via ``sips``, decodes
    HEIC/HEIF to a temporary PNG when Pillow lacks a HEIF plugin. The PNG is
    validated because sips can report success without producing a full image.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"unsupported image format {ext or '(none)'}; supported: {', '.join(sorted(SUPPORTED))}")
    try:
        return _open_pillow(path)
    except (UnidentifiedImageError, OSError) as original_error:
        if ext not in {".heic", ".heif"}:
            raise ValueError(f"cannot decode {path}: {original_error}") from original_error
        if sys.platform != "darwin":
            raise ValueError("HEIC/HEIF requires macOS sips or an installed Pillow HEIF decoder") from original_error
        with tempfile.TemporaryDirectory(prefix="scoredewarp_heic_") as tmp:
            target = Path(tmp) / "photo.png"
            try:
                proc = subprocess.run(["sips", "-s", "format", "png", str(path), "--out", str(target)],
                                      capture_output=True, text=True, timeout=120, check=False)
            except (OSError, subprocess.TimeoutExpired) as e:
                raise ValueError(f"HEIC conversion failed for {path}: {e}") from e
            if proc.returncode != 0:
                raise ValueError(f"HEIC conversion failed for {path}: {proc.stderr.strip() or proc.stdout.strip()}")
            try:
                return _open_pillow(target)
            except (UnidentifiedImageError, OSError) as e:
                raise ValueError(f"HEIC conversion produced an unreadable PNG for {path}: {e}") from e
