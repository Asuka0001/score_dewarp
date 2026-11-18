import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scoredewarp.core.image_input import read_photo


class ImageInputTests(unittest.TestCase):
    def test_common_formats_are_oriented_and_return_bgr(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "photo.webp"
            Image.new("RGB", (20, 10), (255, 0, 0)).save(p, lossless=True)
            a = read_photo(p)
            self.assertEqual(a.shape, (10, 20, 3))
            np.testing.assert_array_equal(a[5, 5], [0, 0, 255])
            p = Path(tmp) / "rotated.jpg"
            im = Image.new("RGB", (20, 10), (255, 0, 0))
            exif = Image.Exif()
            exif[274] = 6
            im.save(p, exif=exif)
            self.assertEqual(read_photo(p).shape, (20, 10, 3))

    def test_bad_formats_and_truncated_image_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wrong.pdf"
            p.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "unsupported image format"):
                read_photo(p)
            p = Path(tmp) / "broken.png"
            p.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "cannot decode"):
                read_photo(p)
