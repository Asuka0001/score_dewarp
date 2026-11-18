import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoredewarp.core.field import _row_ends, margin_stats  # noqa: E402
from scoredewarp.core.structures import Fragment, Staff, Structures, find_fragments, refine_strokes  # noqa: E402


def _tilt(p):
    return np.degrees(np.arctan(np.polyfit(p[:, 1], p[:, 0], 1)[0]))


class StemRefineTests(unittest.TestCase):
    def test_notehead_does_not_tilt_a_vertical_stem(self):
        ink = np.zeros((200, 300), np.uint8)
        for i, x in enumerate(range(40, 280, 30)):
            up = i % 2 == 0
            y0, y1 = (40, 120) if up else (60, 140)
            cv2.line(ink, (x, y0), (x, y1), 1, 2)
            # up-stem: notehead left of the foot; down-stem: right of the top
            c = (x - 6, y1) if up else (x + 6, y0)
            cv2.ellipse(ink, c, (7, 5), -20, 0, 360, 1, -1)
        segs, _ = find_fragments(ink, 2.0, 12.0, horizontal=False, length=26)
        self.assertGreaterEqual(len(segs), 6)
        raw = np.median([_tilt(p) for p in segs])
        ref = refine_strokes(segs, ink, min_span=14)
        self.assertGreaterEqual(len(ref), 6)
        self.assertLess(abs(np.median([_tilt(p) for p in ref])), 0.2)
        self.assertGreater(abs(raw), abs(np.median([_tilt(p) for p in ref])))


def _page(rows, sp=12.0):
    """rows: list of (y_top, [(x0, x1), ...] pieces); one Staff per piece."""
    frags, staves = [], []
    for y, pieces in rows:
        for x0, x1 in pieces:
            ids = []
            for line in range(5):
                yy = y + line * sp
                xs = np.arange(x0, x1 + 1, 6.0)
                frags.append(Fragment(np.stack([xs, np.full(len(xs), yy)], 1), staff=len(staves), line=line))
                ids.append(len(frags) - 1)
            tx = np.arange(x0 + 30, x1 - 30, 40.0)
            tuples = [np.stack([np.full(5, x), y + sp * np.arange(5)], 1) for x in tx]
            staves.append(Staff(ids, tuples, sp, x_range=(x0, x1)))
    return Structures((1200, 900), 2.0, sp, frags, staves, [], [])


class RowEndsTests(unittest.TestCase):
    def test_split_row_and_indent(self):
        rows = [(100, [(150, 800)]),             # indented first system
                (250, [(60, 800)]),
                (400, [(60, 420), (430, 800)]),  # one staff split in two pieces
                (550, [(60, 800)]),
                (700, [(60, 800)])]
        S = _page(rows)
        L = _row_ends(S, 0)
        R = _row_ends(S, -1)
        self.assertEqual(len(L), 4)              # indent rejected
        np.testing.assert_allclose(L[:, 0], 60)  # split row: the left piece's end
        self.assertEqual(len(R), 5)
        np.testing.assert_allclose(R[:, 0], 798, atol=6)

    def test_margin_stats_keystone(self):
        # bottom rows narrower: right ends move left by 20 px per row step
        rows = [(100 + 150 * i, [(60, 800 - 20 * i)]) for i in range(6)]
        ms = margin_stats(_page(rows))
        self.assertAlmostEqual(ms["left"][2], 0.0, places=3)
        self.assertLess(ms["right"][2], 0)
        self.assertLess(ms["keystone"], -0.1)


if __name__ == "__main__":
    unittest.main()
