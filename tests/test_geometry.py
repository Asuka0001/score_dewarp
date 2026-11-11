import unittest
import numpy as np

from scoredewarp.eval.geometry import geometry_metrics, orientation_error_rate
from scoredewarp.core.surface3d import SurfaceModel
from scoredewarp.core.structures import Structures


class GeometryMetricsTests(unittest.TestCase):
    def test_surface_without_structure_allows_coarse_fallback(self):
        structures = Structures((448, 448), 1., 8., [], [], [], [])
        self.assertEqual(SurfaceModel((448, 448)).build(structures, np.eye(3)), 0)

    def setUp(self):
        x, y = np.meshgrid(np.linspace(0, 1, 12), np.linspace(0, 1.4, 16))
        self.gt = np.stack([x, y], -1)
        self.valid = np.ones(x.shape, bool)

    def test_global_similarity_is_removed(self):
        angle = .4
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        pred = 300 * self.gt @ rotation.T + [120, -50]
        metrics = geometry_metrics(self.gt, pred, self.valid)
        self.assertLess(metrics["ad_pct"], 1e-10)
        self.assertLess(metrics["local_distortion_pct"], 1e-10)
        self.assertEqual(metrics["coverage"], 1)

    def test_shear_remains_and_missing_points_reduce_coverage(self):
        pred = self.gt @ np.array([[1, .25], [0, 1]])
        keep = self.valid.copy()
        keep[:8] = False
        metrics = geometry_metrics(self.gt, pred, self.valid, keep)
        self.assertEqual(metrics["coverage"], .5)
        self.assertGreater(metrics["ad_pct"], 1)
        self.assertGreater(metrics["local_distortion_pct"], 1)

    def test_collapsed_or_missing_field_has_no_fake_zero_error(self):
        for pred in (np.zeros_like(self.gt), np.full_like(self.gt, np.nan)):
            metrics = geometry_metrics(self.gt, pred, self.valid)
            self.assertIsNone(metrics["ad_pct"])
            self.assertIsNone(metrics["local_distortion_pct"])

    def test_orientation_check_detects_reflection_and_collapse(self):
        self.assertEqual(orientation_error_rate(self.gt, self.gt, self.valid), 0.)
        self.assertEqual(orientation_error_rate(self.gt, self.gt * [-1, 1], self.valid), 1.)
        self.assertEqual(orientation_error_rate(self.gt, np.zeros_like(self.gt), self.valid), 1.)


if __name__ == "__main__":
    unittest.main()
