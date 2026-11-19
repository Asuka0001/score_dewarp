import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from scoredewarp.core.field import _head_scale, _row_ends
from scoredewarp.core.pipeline import _detect
from scoredewarp.synth.render import render_page


def _page(kind, seed):
    page = render_page(kind, np.random.default_rng(seed))[0]
    return page if page.ndim == 3 else cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


class NoteheadRulerTests(unittest.TestCase):
    def test_noteheads_have_one_width(self):
        S = _detect(_page("staff", 3))[2]
        self.assertGreater(len(S.staves), 2)
        h = S.heads
        self.assertGreater(len(h), 10)
        sp = float(np.median([s.spacing for s in S.staves]))
        self.assertTrue(1.0 < np.median(h[:, 2]) / sp < 1.6)           # a notehead is ~1.2-1.35 spacings wide
        self.assertLess(np.std(h[:, 2]) / np.median(h[:, 2]), 0.2)
        self.assertTrue(set(h[:, 3].astype(int)) <= set(range(len(S.staves))))

    def test_head_scale_sees_horizontal_squeeze(self):
        # right half of the staff squeezed to 80 % width: heads there are 8 px instead of 10
        x = np.arange(20) * 30.0
        w = np.where(x < 300, 10.0, 8.0)
        S = SimpleNamespace(heads=np.stack([x, np.full_like(x, 50), w, np.zeros_like(x)], 1))
        sc = _head_scale(S, 0, np.array([100.0, 500.0]), 10.0)
        self.assertAlmostEqual(sc[0], 1.0, places=6)
        self.assertAlmostEqual(sc[1], 1.25, places=6)
        self.assertIsNone(_head_scale(S, 1, np.array([100.0]), 10.0))  # no heads on that staff


class JianpuBlockSideTests(unittest.TestCase):
    def test_right_side_from_final_barlines_left_side_skipped(self):
        S = _detect(_page("jianpu", 5))[2]
        self.assertFalse(S.staves)
        self.assertEqual(len(_row_ends(S, 0, jianpu=True)), 0)          # first digits do not line up
        R = _row_ends(S, -1, jianpu=True)
        self.assertGreaterEqual(len(R), 3)
        h = float(np.median([r.height for r in S.rows if r.kind == "jianpu"]))
        self.assertLess(np.ptp(R[:, 0]), h)                              # justified rows: one right margin


if __name__ == "__main__":
    unittest.main()
