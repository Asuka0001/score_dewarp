import unittest

import numpy as np

from scoredewarp.core.page import sheet_contour, sheet_sides


def _scene(facing=False):
    """Bright sheet on a dark table; with ``facing`` a bright facing page sits
    just right of a narrow dark gutter (a book spread)."""
    H, W = 1200, 1000
    img = np.full((H, W, 3), 40, np.uint8)
    mask = np.zeros((H, W), np.uint8)
    x0, x1, y0, y1 = 150, 700, 150, 1050
    img[y0:y1, x0:x1] = 225
    mask[y0:y1, x0:x1] = 255
    if facing:
        img[y0:y1, x1 + 12:x1 + 280] = 225     # facing page behind a 12 px gutter shadow
    return img, mask, (x0, x1, y0, y1)


class SheetSidesTest(unittest.TestCase):
    def test_four_sides_on_background(self):
        img, mask, (x0, x1, y0, y1) = _scene()
        c = sheet_contour(mask, img)
        sides = sheet_sides(c, lambda p: np.asarray(p, float), (1000, 1200), block=(250, 600, 250, 950))
        self.assertEqual(set(sides), {"left", "right", "top", "bottom"})
        self.assertLess(np.abs(sides["left"][:, 0] - x0).max(), 3)
        self.assertLess(np.abs(sides["bottom"][:, 1] - y1).max(), 3)

    def test_gutter_with_facing_page_is_not_an_edge(self):
        img, mask, _ = _scene(facing=True)
        c = sheet_contour(mask, img)
        sides = sheet_sides(c, lambda p: np.asarray(p, float), (1000, 1200), block=(250, 600, 250, 950))
        self.assertNotIn("right", sides)
        self.assertIn("left", sides)

    def test_points_beyond_the_block_ends_are_dropped(self):
        img, mask, _ = _scene()
        c = sheet_contour(mask, img)
        # block only in the upper half: the left side is kept only beside it
        sides = sheet_sides(c, lambda p: np.asarray(p, float), (1000, 1200), block=(250, 600, 250, 600))
        self.assertLess(sides["left"][:, 1].max(), 600 + 0.1 * 350 + 1)


if __name__ == "__main__":
    unittest.main()
