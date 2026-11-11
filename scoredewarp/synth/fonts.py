"""Font discovery with fallbacks (macOS paths first, then common Linux paths)."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_DIGIT_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_CJK_FONTS = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
_SYMBOL_FONTS = [
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _existing(paths):
    return [p for p in paths if os.path.exists(p)]


def digit_font_paths():
    return _existing(_DIGIT_FONTS)


def cjk_font_paths():
    return _existing(_CJK_FONTS)


def symbol_font_paths():
    return _existing(_SYMBOL_FONTS)


@lru_cache(maxsize=256)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=4096)
def ink_bbox(path: str, size: int, text: str, anchor: str = "ls"):
    """Tight ink bbox (x0, y0, x1, y1) of `text` drawn at origin (0,0) with `anchor`.

    PIL's getbbox returns the layout box, not the ink; we need the ink box for
    accurate ground-truth character geometry.
    """
    font = load_font(path, size)
    pad = size * 2
    img = Image.new("L", (pad * 2 + size * len(text) * 2, pad * 2), 0)
    ImageDraw.Draw(img).text((pad, pad), text, font=font, fill=255, anchor=anchor)
    a = np.asarray(img)
    ys, xs = np.nonzero(a > 127)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (xs.min() - pad, ys.min() - pad, xs.max() + 1 - pad, ys.max() + 1 - pad)
