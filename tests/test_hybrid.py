import unittest
from unittest.mock import patch

import numpy as np
import torch

from scoredewarp.core.pipeline import Config, refine_dewarp
from scoredewarp.core.transforms import BackwardMap
from scoredewarp.learn.infer import backward_map


class HybridTests(unittest.TestCase):
    def test_learned_identity_preserves_photo_coordinates(self):
        yy, xx = np.mgrid[:40, :32]
        out = {"fwd": torch.tensor(np.stack([xx / 32, yy / 40]), dtype=torch.float32)[None],
               "page": torch.full((1, 1, 40, 32), 10.)}
        bm = backward_map(out, (320, 400), (320, 400))
        points = np.array([[48, 64], [144, 200], [280, 320]], dtype=float)
        np.testing.assert_allclose(bm.sample(points), points, atol=1e-4)
        np.testing.assert_allclose(bm.forward_points(points), points, atol=1e-4)

    def test_empty_and_collapsed_predictions_fail_explicitly(self):
        out = {"fwd": torch.full((1, 2, 16, 16), .5), "page": torch.full((1, 1, 16, 16), -10.)}
        with self.assertRaisesRegex(ValueError, "Not enough"):
            backward_map(out, (64, 64), (64, 64))
        out["page"].fill_(10.)
        with self.assertRaisesRegex(ValueError, "collapsed"):
            backward_map(out, (64, 64), (64, 64))

    def test_refinement_composes_to_original_photo(self):
        # Learned map samples x+8; a residual samples x+4. The result must sample
        # x+12 in the original photo, rather than double-warping an intermediate.
        photo = np.random.default_rng(7).integers(0, 256, (80, 80, 3), dtype=np.uint8)
        initial = BackwardMap.from_function(lambda p: p + [8, 0], (64, 64))
        residual = BackwardMap.from_function(lambda p: p + [4, 0], (64, 64))
        with patch("scoredewarp.core.pipeline.fit_field", return_value=residual):
            result = refine_dewarp(photo, initial, Config(field_iters=1))
        np.testing.assert_allclose(result.bmap.sample([[16, 16]]), [[28, 16]], atol=1e-5)
        np.testing.assert_array_equal(result.image, result.bmap.apply(photo))
        self.assertTrue(result.stages["surface_skipped"])

    def test_no_structures_keeps_learned_map(self):
        photo = np.full((80, 80, 3), 255, np.uint8)
        initial = BackwardMap.from_homography(np.eye(3), (80, 80))
        result = refine_dewarp(photo, initial)
        self.assertIs(result.bmap, initial)
        np.testing.assert_array_equal(result.image, photo)

    def test_faint_staff_recovery_updates_exports_without_warping_identity(self):
        import cv2
        photo = np.full((360, 600, 3), 255, np.uint8)
        for i in range(5):
            cv2.line(photo, (30, 60 + 10 * i), (570, 60 + 10 * i), (220, 220, 220), 2)
        for x in range(80, 550, 70):
            cv2.ellipse(photo, (x, 85), (7, 5), 0, 0, 360, (20, 20, 20), -1)
        initial = BackwardMap.from_homography(np.eye(3), (600, 360))
        result = refine_dewarp(photo, initial, Config(use_field=False))
        self.assertIs(result.bmap, initial)
        self.assertEqual(result.page_type, 'staff')
        self.assertEqual(len(result.structures.staves), 1)
        np.testing.assert_array_equal(result.image, photo)


if __name__ == "__main__":
    unittest.main()
