import unittest

import cv2
import numpy as np

from scoredewarp.core import glyphnet
from scoredewarp.core.metric_pipeline import MetricConfig, dewarp_metric
from scoredewarp.core.pipeline import _detect, drop_rim
from scoredewarp.synth.render import render_page

from test_pipeline_smoke import _photo


class DropRimTests(unittest.TestCase):
    def test_border_sliver_goes_content_and_frame_edge_stay(self):
        norm = np.full((300, 400), 230, np.uint8)
        valid = np.zeros_like(norm)
        valid[20:280, 0:370] = 1                 # left side reaches the photo frame
        norm[20:24, 40:340] = 40                 # dark background sliver along the top mask border
        norm[150:153, 100:300] = 40              # a printed line inside the sheet
        norm[120:200, 0:3] = 40                  # ink cut by the photo frame
        out = drop_rim(norm, valid)
        self.assertEqual(out[21, 200], 0)
        self.assertEqual(out[151, 200], 1)
        self.assertEqual(out[150, 1], 1)
        self.assertEqual(out[100, 200], 1)       # the sheet itself is untouched


class DigitFirstRowsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        page = render_page("jianpu", np.random.default_rng(5))[0]
        page = page if page.ndim == 3 else cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
        cls.img = cv2.resize(page, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

    def _jp_rows(self, **kw):
        S = _detect(self.img, **kw)[2]
        return [r for r in S.rows if r.kind == "jianpu"]

    def test_digit_first_finds_note_rows(self):
        base = self._jp_rows()
        for glyph in ("template", "cnn"):
            if glyph == "cnn" and not glyphnet.available():
                continue
            with self.subTest(glyph=glyph):
                rows = self._jp_rows(jp_rows="digit", glyph=glyph)
                self.assertGreater(len(rows), 0)
                self.assertGreaterEqual(sum(len(r.full) for r in rows), 0.8 * sum(len(r.full) for r in base))

    def test_glyph_cnn_reads_rendered_digits(self):
        if not glyphnet.available():
            self.skipTest("glyph model not available")
        crops = []
        for d in "1234567":
            im = np.zeros((40, 30), np.uint8)
            cv2.putText(im, d, (5, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 1, 3)
            ys, xs = np.nonzero(im)
            crops.append(im[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
        got = [c for c, _ in glyphnet.classify(crops, thr=0.5)]
        self.assertGreaterEqual(sum(g == d for g, d in zip(got, "1234567")), 5)


class WidthPoolTests(unittest.TestCase):
    def test_keeps_least_field_work(self):
        img = _photo()
        res = dewarp_metric(img, MetricConfig(output_res="canvas", pool="width", pool_widths=(1000, 990)))
        pool = res.stages["pool"]
        self.assertEqual(len(pool["candidates"]), 2)
        disp = [np.inf if c["disp"] is None else c["disp"] for c in pool["candidates"]]
        self.assertEqual(pool["selected"], int(np.argmin(disp)))
        self.assertGreater(len(res.structures.staves), 0)

    def test_unknown_mode_fails(self):
        with self.assertRaises(ValueError):
            dewarp_metric(_photo(), MetricConfig(pool="on"))


if __name__ == "__main__":
    unittest.main()
