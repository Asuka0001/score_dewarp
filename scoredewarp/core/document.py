"""Physical paper layout, independent of the geometric dewarping model."""
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


PAPER_MM = {"a4": (210.0, 297.0), "a3": (297.0, 420.0), "letter": (215.9, 279.4)}


@dataclass(frozen=True)
class DocumentConfig:
    paper: str = "a4"
    orientation: str = "portrait"
    dpi: int = 300
    margin_mm: float = 10.0
    size_mm: tuple | None = None

    def __post_init__(self):
        if self.paper not in (*PAPER_MM, "custom", "adaptive"):
            raise ValueError("unknown paper size")
        if self.orientation not in ("portrait", "landscape"):
            raise ValueError("orientation must be portrait or landscape")
        if not 36 <= self.dpi <= 1200:
            raise ValueError("dpi must be between 36 and 1200")
        if not np.isfinite(self.margin_mm) or self.margin_mm < 0:
            raise ValueError("margin-mm must be finite and nonnegative")
        if self.paper == "custom":
            if self.size_mm is None or len(self.size_mm) != 2:
                raise ValueError("custom paper requires --size-mm WIDTH HEIGHT")
            if not all(np.isfinite(v) and v > 0 for v in self.size_mm):
                raise ValueError("paper dimensions must be finite and positive")
        elif self.size_mm is not None:
            raise ValueError("--size-mm requires --paper custom")
        if self.paper != "adaptive":
            w, h = self.physical_size()
            if 2 * self.margin_mm >= min(w, h):
                raise ValueError("margins leave no printable area")
            px = [round(v * self.dpi / 25.4) for v in (w, h)]
            if min(px) < 1 or px[0] * px[1] > 80_000_000 or max(px) >= 32767:
                raise ValueError("paper size / dpi exceeds supported raster size (80 MP, side <32767)")

    def physical_size(self):
        w, h = self.size_mm if self.paper == "custom" else PAPER_MM[self.paper]
        w, h = sorted((w, h))
        return (h, w) if self.orientation == "landscape" else (w, h)


@dataclass
class DocumentLayout:
    config: DocumentConfig
    source_size: tuple
    out_size: tuple
    matrix: np.ndarray  # rectified pixel centres -> document pixel centres

    @classmethod
    def fit(cls, source_size, config):
        w, h = source_size
        if min(w, h) <= 0:
            raise ValueError("source size must be positive")
        if config.paper == "adaptive":
            return cls(config, (w, h), (w, h), np.eye(3))
        ow, oh = (round(v * config.dpi / 25.4) for v in config.physical_size())
        margin = int(np.ceil(config.margin_mm * config.dpi / 25.4))
        scale = min((ow - 2 * margin) / w, (oh - 2 * margin) / h)
        if scale <= 0:
            raise ValueError("margins leave no printable pixels at this dpi")
        # Align pixel boundaries, including the complete first / last pixel.
        tx, ty = (ow - scale * w + scale - 1) / 2, (oh - scale * h + scale - 1) / 2
        matrix = np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1.]])
        return cls(config, (w, h), (ow, oh), matrix)

    def render(self, image):
        """Fit a prepared reading / binary image without stretching or cropping."""
        if image.shape[1::-1] != self.source_size:
            raise ValueError("image size does not match rectified coordinates")
        if self.config.paper == "adaptive":
            return image.copy()
        fill = (255,) * image.shape[2] if image.ndim == 3 else 255
        return cv2.warpAffine(image, self.matrix[:2], self.out_size,
                              flags=cv2.INTER_LINEAR, borderValue=fill)

    def render_mask(self, mask):
        """Map a rectified paper mask to the page without blurring its edge."""
        if mask is None:
            return None
        if mask.shape[::-1] != self.source_size:
            raise ValueError("mask size does not match rectified coordinates")
        if self.config.paper == "adaptive":
            return mask.copy()
        return cv2.warpAffine(mask, self.matrix[:2], self.out_size,
                              flags=cv2.INTER_NEAREST, borderValue=0)

    def render_photo(self, photo, bmap):
        """Compose layout with the geometry and sample the original photo once."""
        if bmap.out_size != self.source_size:
            raise ValueError("map size does not match rectified coordinates")
        if self.config.paper == "adaptive":
            return bmap.apply(photo, value=255)
        mx, my = bmap.full_maps()
        # Replication avoids inventing source coordinates at the canvas edge;
        # an explicit footprint below makes all added paper white.
        maps = [cv2.warpAffine(m, self.matrix[:2], self.out_size,
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for m in (mx, my)]
        out = cv2.remap(photo, *maps, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255) if photo.ndim == 3 else 255)
        mask = cv2.warpAffine(np.ones(self.source_size[::-1], np.uint8), self.matrix[:2],
                              self.out_size, flags=cv2.INTER_NEAREST, borderValue=0)
        out[mask == 0] = 255
        return out

    def metadata(self):
        c = self.config
        mm = c.physical_size() if c.paper != "adaptive" else tuple(v * 25.4 / c.dpi for v in self.out_size)
        return {"paper": c.paper, "orientation": c.orientation if c.paper != "adaptive" else None,
                "size_mm": list(mm), "dpi": c.dpi, "size_px": list(self.out_size),
                "margin_mm": c.margin_mm if c.paper != "adaptive" else None,
                "fit": "contain", "source_size_px": list(self.source_size),
                "rectified_to_document": self.matrix.tolist(),
                "coordinates": "structures and bmap remain in rectified image coordinates"}


def save_document(base, reading, color, layout, binary=None, pdf=True):
    """Export PNG with DPI, a reading PDF and a small visual preview."""
    dpi = layout.config.dpi
    def pil(a):
        return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if a.ndim == 3 else a)
    pil(reading).save(base + "_page.png", dpi=(dpi, dpi))
    pil(color).save(base + "_page_color.png", dpi=(dpi, dpi))
    if binary is not None:
        pil(binary).convert("1", dither=Image.Dither.NONE).save(base + "_page_binary.png", dpi=(dpi, dpi))
    if pdf:
        pil(reading).convert("RGB").save(base + "_page.pdf", resolution=dpi, quality=95)
    preview = pil(reading).convert("RGB")
    preview.thumbnail((1000, 1000))
    preview.save(base + "_preview.jpg", quality=90)
