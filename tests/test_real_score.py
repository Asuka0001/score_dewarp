import unittest
from types import SimpleNamespace
import numpy as np

from scoredewarp.eval.real_score import trace_metrics, sampled_map_folds, margin_content_points


class RealScoreTests(unittest.TestCase):
    def setUp(self):
        x = np.linspace(10, 210, 21)
        self.staff = np.stack([np.stack([x, np.full_like(x, 30+10*i)], -1) for i in range(5)], 1)
        self.valid = np.ones((21, 5), bool)

    def test_scale_invariance_and_bending_error(self):
        bent = self.staff.copy()
        bent[:, :, 1] += (.002*(bent[:, :, 0]-110)**2)
        a = trace_metrics([bent], [self.valid])
        b = trace_metrics([bent*3+[100, 200]], [self.valid])
        self.assertGreater(a['staff_straight'], .1)
        self.assertAlmostEqual(a['staff_straight'], b['staff_straight'])
        self.assertEqual(trace_metrics([self.staff], [self.valid])['staff_straight'], 0.)

    def test_missing_and_collapsed_traces_do_not_get_perfect_scores(self):
        valid = self.valid.copy(); valid[:15] = False
        result = trace_metrics([self.staff], [valid])
        self.assertIsNone(result['staff_straight'])
        self.assertAlmostEqual(result['point_coverage'], 6/21)
        self.assertIsNone(trace_metrics([np.zeros_like(self.staff)], [self.valid])['staff_straight'])
        self.assertIsNone(trace_metrics([], [])['point_coverage'])

    def test_common_support_prevents_cropping_from_improving_own_comparison(self):
        bent = self.staff.copy(); bent[:5, :, 1] += 50
        crop = self.valid.copy(); crop[:5] = False
        own = trace_metrics([bent], [self.valid])
        common = trace_metrics([bent], [crop])
        self.assertGreater(own['staff_straight'], common['staff_straight'])
        self.assertLess(common['point_coverage'], 1.)

    def test_margin_content_points_cover_header_and_footer_bands_only(self):
        staves = [self.staff + [0, 60*k] for k in range(3)]   # tops 30, 90, 150; gap 60
        ink = np.zeros((300, 240), np.uint8)
        ink[5:10, 50:60] = 1      # header in the band above the first staff
        ink[40:60, 50:60] = 1     # inside staves: excluded
        ink[230:235, 50:60] = 1   # footer in the band below the last staff
        ink[280:290, 50:60] = 1   # beyond one gap below: excluded
        points = margin_content_points(ink, staves, stride=1)
        ys = points[:, 1]
        self.assertEqual(len(points), 100)
        self.assertTrue(((ys >= 5) & (ys < 10) | (ys >= 230) & (ys < 235)).all())
        self.assertEqual(len(margin_content_points(ink, staves[:1])), 0)

    def test_reversals_and_sampled_folds(self):
        reversed_x = self.staff.copy(); reversed_x[:, :, 0] *= -1
        self.assertEqual(trace_metrics([reversed_x], [self.valid])['horizontal_reversal_rate'], 1.)
        x, y = np.meshgrid(np.arange(5), np.arange(6))
        grid = np.stack([x, y], -1).astype(float)
        self.assertEqual(sampled_map_folds(SimpleNamespace(grid=grid))['nonpositive_jacobian_fraction'], 0.)
        self.assertEqual(sampled_map_folds(SimpleNamespace(grid=grid*[-1, 1]))['nonpositive_jacobian_fraction'], 1.)
        grid[:] = np.nan
        self.assertIsNone(sampled_map_folds(SimpleNamespace(grid=grid))['nonpositive_jacobian_fraction'])


if __name__ == '__main__':
    unittest.main()
