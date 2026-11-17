import unittest

import cv2
import numpy as np

from scoredewarp.core.page import page_edges
from scoredewarp.core.preprocess import clean_page, protect_content
from scoredewarp.core.structures import _merge_chains


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


    def test_mask_never_cuts_content(self):
        H, W = 600, 500
        img = np.full((H, W, 3), 230, np.uint8)
        for y in range(480, 530, 12):                # last staff, cut by the mask
            img[y:y + 2, 40:460] = 30
        for x in range(60, 440, 30):
            cv2.circle(img, (x, 498), 6, (20, 20, 20), -1)
        rng = np.random.default_rng(0)
        speck = rng.random((H, 120)) < 0.02          # sparse cloth texture
        img[:, 380:][speck[:, :120]] = 90
        region = np.zeros((H, W), np.uint8)
        region[:505, :380] = 255                     # sheet mask: misses the lower half of the staff
        out = protect_content(region, img, spacing=12)
        self.assertTrue((out[480:530, 60:370] > 0).all())     # whole staff kept
        self.assertTrue((out[:400, 400:] == 0).mean() > 0.95)  # texture stays off-sheet


class MergeRowsTest(unittest.TestCase):
    def test_wide_gap_joined_lyrics_kept_apart(self):
        h = 26.0
        boxes = []
        for x in [100, 140, 180]:                    # "5 4 4" left piece, a bit higher
            boxes.append([x, 300 - 15, x + 16, 300 - 15 + h])
        for x in range(420, 900, 40):                # rest of the row after "- - |"
            boxes.append([x, 300, x + 16, 300 + h])
        for x in range(100, 900, 40):                # lyric row 2.5 heights below
            boxes.append([x, 300 + 2.5 * h, x + 24, 300 + 3.5 * h])
        B = np.array(boxes, float)
        n_left, n_right = 3, len(range(420, 900, 40))
        chains = [list(range(n_left)), list(range(n_left, n_left + n_right)),
                  list(range(n_left + n_right, len(B)))]
        out = _merge_chains(chains, B)
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(len(c) for c in out), sorted([n_left + n_right, len(B) - n_left - n_right]))


if __name__ == "__main__":
    unittest.main()
