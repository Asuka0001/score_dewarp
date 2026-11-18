import unittest

import cv2
import numpy as np

from scoredewarp.core.metric_pipeline import MetricConfig, dewarp_metric, method_from_spec
from scoredewarp.synth.render import render_page


def _photo(kind="staff", seed=3):
    """Synthetic page on a dark, textured table under a mild perspective."""
    page = render_page(kind, np.random.default_rng(seed))
    page = page[0] if isinstance(page, tuple) else page
    page = page if page.ndim == 3 else cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    page = cv2.resize(page, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    h, w = page.shape[:2]
    W, H = int(w * 1.4), int(h * 1.3)
    rng = np.random.default_rng(seed)
    bg = (60 + 30 * rng.random((H // 8 + 1, W // 8 + 1, 3))).astype(np.uint8)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_NEAREST)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0.18 * W, 0.10 * H], [0.84 * W, 0.12 * H], [0.88 * W, 0.90 * H], [0.12 * W, 0.88 * H]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, M, (W, H), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H)) > 127
    out = bg.copy()
    out[mask] = warped[mask]
    return out


class PipelineSmokeTest(unittest.TestCase):
    """End to end on a small synthetic photo: every option combination runs."""

    @classmethod
    def setUpClass(cls):
        cls.img = _photo()

    def test_default(self):
        res = dewarp_metric(self.img, MetricConfig(output_res="canvas"))
        self.assertGreater(res.image.shape[0], 100)
        self.assertGreater(len(res.structures.staves), 0)

    def test_model_options(self):
        from scoredewarp.core import learned, pagenet
        if not (learned.available() and pagenet.available()):
            self.skipTest("models not available")
        for spec in ("gcs:extent_fold=true",
                     "gcs:page_seg=u2netp,page_quad=true,extent_fold=true,page_erode=0.002,b_select=outline",
                     "gcs:page_seg=u2netp,edge_prefit=true",
                     "gcs:seg=homr,seg_parts=staff,seg_fuse=staff,page_seg=u2netp"):
            with self.subTest(spec=spec):
                res = method_from_spec(spec)(self.img)
                self.assertGreater(res.image.shape[0], 100)


if __name__ == "__main__":
    unittest.main()
