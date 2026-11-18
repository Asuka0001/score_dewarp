import unittest

import cv2
import numpy as np

from scoredewarp.core.learned import CHANNELS, masks
from scoredewarp.core.pipeline import _warp_seg
from scoredewarp.core.structures import detect_structures
from scoredewarp.core.transforms import BackwardMap


def _page(n_staves=3, W=900, H=700, sp=12):
    img = np.full((H, W), 255, np.uint8)
    staff = np.zeros((H, W), np.uint8)
    for k in range(n_staves):
        y0 = 100 + 200 * k
        for i in range(5):
            y = y0 + i * sp
            cv2.line(img, (60, y), (W - 60, y), 0, 1)
            cv2.line(staff, (60, y), (W - 60, y), 1, 2)
        for x in range(120, W - 80, 70):          # stems with noteheads
            cv2.line(img, (x, y0 - 10), (x, y0 + 3 * sp), 0, 1)
            cv2.ellipse(img, (x - 5, y0 + 3 * sp), (6, 4), -20, 0, 360, 0, -1)
    return img, staff


class LearnedSegTest(unittest.TestCase):
    def test_masks_threshold_and_resize(self):
        seg = np.zeros((10, 20, len(CHANNELS)), np.uint8)
        seg[2, 3, 0] = 200
        m = masks(seg)
        self.assertEqual(set(m), set(CHANNELS))
        self.assertEqual(m["staff"][2, 3], 1)
        self.assertEqual(m["stem"].sum(), 0)
        self.assertEqual(masks(seg, (20, 40))["staff"].shape, (20, 40))
        self.assertIsNone(masks(None))

    def test_staves_from_mask(self):
        img, staff = _page()
        ink = (img < 128).astype(np.uint8)
        seg = {c: np.zeros_like(staff) for c in CHANNELS}
        seg["staff"] = staff
        S = detect_structures(img, ink, robust=False, seg=seg)
        self.assertEqual(len(S.staves), 3)
        self.assertAlmostEqual(S.staves[0].spacing, 12, delta=1.0)

    def test_empty_mask_falls_back(self):
        img, _ = _page()
        ink = (img < 128).astype(np.uint8)
        seg = {c: np.zeros_like(ink) for c in CHANNELS}
        a = detect_structures(img, ink, robust=False)
        b = detect_structures(img, ink, robust=False, seg=seg)
        self.assertEqual(a.summary(), b.summary())

    def test_warp_seg_many_channels(self):
        seg = np.random.default_rng(0).integers(0, 255, (50, 60, 5), dtype=np.uint8)
        bm = BackwardMap.from_homography(np.eye(3), (60, 50))
        out = _warp_seg(bm, seg)
        self.assertEqual(out.shape, seg.shape)
        self.assertLess(np.abs(out.astype(int) - seg).max(), 2)
        self.assertIsNone(_warp_seg(bm, None))


class BundledModelTest(unittest.TestCase):
    def test_bundled_model_finds_staff_lines(self):
        import os
        from scoredewarp.core import learned
        if not learned.available():
            self.skipTest("onnxruntime or bundled model missing")
        os.environ["SCOREDEWARP_SEG_COREML"] = "0"     # plain CPU in tests
        img, staff = _page(n_staves=2, W=640, H=480, sp=16)
        seg = learned.segment(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), width=640)
        self.assertEqual(seg.shape, (480, 640, len(CHANNELS)))
        pred = seg[..., CHANNELS.index("staff")] > 127
        near = cv2.dilate(staff, np.ones((5, 5), np.uint8)) > 0
        self.assertGreater(pred[near].mean(), 0.3)       # lines are found ...
        self.assertLess(pred[~near].mean(), 0.02)        # ... and little else


if __name__ == "__main__":
    unittest.main()
