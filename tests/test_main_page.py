import unittest

import cv2
import numpy as np

from scoredewarp.core.mainpage import main_page
from scoredewarp.core.pipeline import Result
from scoredewarp.core.structures import Staff, Structures, crop_structures
from scoredewarp.core.transforms import BackwardMap

SP = 12.0


def _staff(x0, x1, y):
    xs = np.linspace(x0, x1, 8)
    tuples = [np.array([[x, y + i * SP] for i in range(5)], float) for x in xs]
    return Staff(frag_ids=[], tuples=tuples, spacing=SP, top_poly=np.array([0.0, float(y)]),
                 bot_poly=np.array([0.0, y + 4 * SP]), x_range=(float(x0), float(x1)))


def _draw(img, st):
    for i in range(5):
        y = int(round(st.tuples[0][i, 1]))
        cv2.line(img, (int(st.x_range[0]), y), (int(st.x_range[1]), y), (30, 30, 30), 2)
    for x in np.linspace(st.x_range[0] + 20, st.x_range[1] - 20, 12):   # noteheads
        cv2.circle(img, (int(x), int(st.tuples[0][2, 1])), 6, (30, 30, 30), -1)


def _scene(facing=True, columns=False):
    """Main page (white sheet, 8 staves, a page number above, a dark table
    below the sheet) and, with ``facing``, a strip of facing page with two
    short staves right of a gutter. ``columns``: the page holds two balanced
    columns of staves instead (a two-column sheet: nothing may be cut)."""
    H, W = 1400, 1400
    img = np.full((H, W, 3), 60, np.uint8)
    paper = np.zeros((H, W), np.uint8)
    img[60:1120, 80:1040] = 235     # sheet ends right below the last staff (1088)
    paper[60:1120, 80:1040] = 255
    staves = []
    if columns:
        for k in range(6):
            staves += [_staff(150, 520, 200 + 150 * k), _staff(600, 970, 200 + 150 * k)]
    else:
        staves = [_staff(150, 970, 200 + 120 * k) for k in range(8)]
    cv2.putText(img, "16", (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
    if facing:
        img[60:1250, 1110:1400] = 235
        paper[60:1250, 1110:1400] = 255
        staves += [_staff(1150, 1390, 300), _staff(1150, 1390, 500)]
    for st in staves:
        _draw(img, st)
    S = Structures(shape=(H, W), thickness=2.0, spacing_est=SP, fragments=[], staves=staves, rows=[], vsegs=[],
                   page_type="staff")
    bm = BackwardMap.from_homography(np.eye(3), (W, H))
    return Result(img.copy(), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), bm, S, "staff", {}, {}, paper=paper)


class MainPageTest(unittest.TestCase):
    def test_facing_page_and_background_removed(self):
        res = _scene()
        out = main_page(res)
        info = out.stages["main_page"]
        x0, y0, x1, y1 = info["box"]
        self.assertIn("gutter", info)
        self.assertLess(x1, 1110)                 # facing page outside the box
        self.assertLess(x0, 150)                  # clefs kept with a margin
        self.assertLess(y0, 115)                  # page number above the first staff kept
        self.assertGreater(y1, 200 + 7 * 120 + 4 * SP)
        self.assertEqual(out.image.shape[:2], (y1 - y0, x1 - x0))
        self.assertEqual(out.bmap.out_size, (x1 - x0, y1 - y0))
        np.testing.assert_allclose(out.bmap.sample([[0, 0]])[0], [x0, y0], atol=1e-3)
        self.assertEqual(len(out.structures.staves), 8)
        np.testing.assert_allclose(out.structures.staves[0].tuples[0][0], [150 - x0, 200 - y0])
        # the dark table below the sheet, inside the margin, is whitened; print kept
        self.assertGreater(y1, 1130)
        self.assertGreater(out.image[1130 - y0:, 300 - x0:900 - x0].min(), 250)
        self.assertLess(out.image[200 - y0:250 - y0, 150 - x0:970 - x0].min(), 60)

    def test_two_columns_kept(self):
        out = main_page(_scene(facing=False, columns=True))
        info = out.stages["main_page"]
        self.assertNotIn("gutter", info)
        x0, _, x1, _ = info["box"]
        self.assertLess(x0, 150)
        self.assertGreater(x1, 970)


class CropStructuresTest(unittest.TestCase):
    def test_poly_and_drop(self):
        st = _staff(100, 500, 300)
        st.top_poly = np.array([1e-4, 0.02, 290.0])   # y = f(x), highest power first
        far = _staff(900, 1000, 300)
        S = Structures(shape=(1000, 1200), thickness=2.0, spacing_est=SP, fragments=[], staves=[st, far], rows=[],
                       vsegs=[], heads=np.array([[200.0, 324.0, 10.0, 0], [950.0, 324.0, 10.0, 1]]))
        T = crop_structures(S, 50, 100, 800, 900)
        self.assertEqual(len(T.staves), 1)
        self.assertEqual(T.shape, (800, 750))
        for x in (60.0, 250.0, 400.0):   # same curve in the crop's pixels
            self.assertAlmostEqual(np.polyval(T.staves[0].top_poly, x - 50), np.polyval(st.top_poly, x) - 100)
        np.testing.assert_allclose(T.heads, [[150.0, 224.0, 10.0, 0]])


if __name__ == "__main__":
    unittest.main()
