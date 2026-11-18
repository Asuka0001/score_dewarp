import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scoredewarp.core.document import DocumentConfig, DocumentLayout, save_document
from scoredewarp.core.transforms import BackwardMap


class DocumentTests(unittest.TestCase):
    def test_a4_contains_full_canvas_and_centres_without_stretch(self):
        layout = DocumentLayout.fit((1700, 2600), DocumentConfig())
        self.assertEqual(layout.out_size, (2480, 3508))
        m = layout.matrix
        self.assertEqual(m[0, 0], m[1, 1])
        bounds = np.array([[-.5, -.5, 1], [1699.5, 2599.5, 1]]) @ m.T
        left = bounds[0, :2] + .5
        right = np.array(layout.out_size) - .5 - bounds[1, :2]
        np.testing.assert_allclose(left, right)
        self.assertTrue((left >= 10 * 300 / 25.4).all())

    def test_landscape_custom_and_adaptive(self):
        self.assertEqual(DocumentLayout.fit((500, 800), DocumentConfig(orientation="landscape")).out_size,
                         (3508, 2480))
        c = DocumentConfig(paper="custom", size_mm=(100, 200), dpi=254)
        self.assertEqual(DocumentLayout.fit((500, 800), c).out_size, (1000, 2000))
        a = np.arange(20, dtype=np.uint8).reshape(4, 5)
        layout = DocumentLayout.fit((5, 4), DocumentConfig(paper="adaptive"))
        np.testing.assert_array_equal(layout.render(a), a)

    def test_reject_invalid_print_settings(self):
        for kw in ({"dpi": 0}, {"margin_mm": -1}, {"margin_mm": 105},
                   {"paper": "custom"}, {"size_mm": (100, 200)},
                   {"paper": "custom", "size_mm": (float("nan"), 200)},
                   {"paper": "custom", "size_mm": (100000, 100000)}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                DocumentConfig(**kw)

    def test_photo_composition_preserves_corners_and_white_margins(self):
        # Coloured corner squares detect cropping, channel changes and misplaced padding.
        photo = np.full((80, 100, 3), 100, np.uint8)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)]
        centres = [(4, 4), (95, 4), (4, 75), (95, 75)]
        for (x, y), color in zip(centres, colors):
            photo[max(y-4, 0):y+5, max(x-4, 0):x+5] = color
        bm = BackwardMap.from_homography(np.eye(3), (100, 80))
        layout = DocumentLayout.fit((100, 80), DocumentConfig(dpi=72))
        out = layout.render_photo(photo, bm)
        self.assertTrue((out[:20] == 255).all())
        for (x, y), color in zip(centres, colors):
            q = np.rint(layout.matrix @ [x, y, 1]).astype(int)
            np.testing.assert_array_equal(out[q[1], q[0]], color)

    def test_png_dpi_pdf_size_and_preview(self):
        layout = DocumentLayout.fit((100, 80), DocumentConfig(dpi=72))
        a = layout.render(np.zeros((80, 100), np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "score")
            save_document(base, a, a, layout, binary=a)
            with Image.open(base + "_page.png") as im:
                self.assertEqual(im.size, layout.out_size)
                self.assertAlmostEqual(im.info["dpi"][0], 72, places=1)
            pdf = Path(base + "_page.pdf").read_bytes()
            box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
            self.assertIsNotNone(box)
            self.assertAlmostEqual(float(box[1]) * 25.4 / 72, 210, delta=.2)
            self.assertAlmostEqual(float(box[2]) * 25.4 / 72, 297, delta=.2)
            with Image.open(base + "_page_binary.png") as im:
                self.assertEqual(im.mode, "1")
            self.assertTrue(Path(base + "_preview.jpg").is_file())
