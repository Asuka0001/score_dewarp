import unittest

import cv2
import numpy as np

from scoredewarp.core.page import page_edges
from scoredewarp.core.preprocess import clean_page


class PageEdgesTest(unittest.TestCase):
    def test_rectangle_sides(self):
        mask = np.zeros((1000, 800), np.uint8)
        mask[100:900, 120:700] = 255
        e = page_edges(mask, np.eye(3))
        self.assertEqual(set(e), {"top", "right", "bottom", "left"})
        self.assertLess(np.abs(e["top"][:, 1] - 100).max(), 4)
        self.assertLess(np.abs(e["bottom"][:, 1] - 899).max(), 4)
        self.assertLess(np.abs(e["left"][:, 0] - 120).max(), 4)
        self.assertLess(np.abs(e["right"][:, 0] - 699).max(), 4)

    def test_side_on_photo_frame_is_dropped(self):
        mask = np.zeros((1000, 800), np.uint8)
        mask[100:1000, 120:700] = 255  # sheet runs out of the bottom of the photo
        e = page_edges(mask, np.eye(3))
        self.assertNotIn("bottom", e)
        self.assertIn("top", e)


class CleanPageTest(unittest.TestCase):
    def test_shadow_removed_ink_kept(self):
        H, W = 600, 500
        page = np.full((H, W), 235, np.float32)
        for y in range(100, 500, 12):
            page[y:y + 2, 50:450] = 30           # staff-like lines
        cv2.circle(page, (250, 300), 7, 20, -1)  # notehead
        shade = np.ones((H, W), np.float32)
        shade[:, 250:] = 0.5                      # sharp cast shadow
        img = np.clip(page * shade, 0, 255).astype(np.uint8)
        out = clean_page(img, spacing=12)
        paper = out[106:110, 60:440]
        self.assertGreater(paper[:, :180].mean(), 240)
        self.assertGreater(paper[:, 210:].mean(), 240)  # shadowed paper is white too
        self.assertLess(out[100:102, 300:440].mean(), 100)  # lines in the shadow stay dark
        self.assertLess(out[300, 250], 100)


if __name__ == "__main__":
    unittest.main()
