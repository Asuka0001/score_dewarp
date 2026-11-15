import unittest

import cv2
import numpy as np

from scoredewarp.learn.structure_postprocess import ridge_scores, structure_masks, threshold_counts
from scoredewarp.eval.auxiliary import structure_count_metrics


class StructurePostprocessTests(unittest.TestCase):
    def test_axes_plateaus_boundaries_and_input_preservation(self):
        p = np.zeros((3, 9, 11), np.float32)
        p[0, 2:5, :] = .8
        p[1, 0, :] = .9
        p[2, :, 6:9] = .7
        original = p.copy()
        result = ridge_scores(p, 2)
        np.testing.assert_array_equal(p, original)
        self.assertEqual(np.count_nonzero(result[0]), 11)
        self.assertTrue(np.all(result[0, 4] == .8))
        self.assertTrue(np.all(result[1, 0] == .9))
        self.assertEqual(np.count_nonzero(result[2]), 9)
        self.assertTrue(np.all(result[2, :, 8] == .7))
        np.testing.assert_array_equal(ridge_scores(p, 0), p)

    def test_histograms_match_brute_force_with_ties_and_absent_channels(self):
        rng = np.random.default_rng(3)
        scores = rng.integers(0, 5, (3, 13, 17)).astype(np.float32) / 4
        truth = rng.random(scores.shape) > .8
        truth[1] = False
        scores[2] = 0
        grid = [.25, .5, .75, 1.]
        for radius in (0, 2):
            p = ridge_scores(scores, radius)
            for tolerance in (0, 1, 2):
                counts = threshold_counts(p, truth, grid, tolerance)
                kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
                for i, threshold in enumerate(grid):
                    mask = structure_masks(scores, [threshold]*3, radius)
                    near_gt = np.stack([cv2.dilate(t.astype(np.uint8), kernel)>0 for t in truth])
                    near_pred = np.stack([cv2.dilate(t.astype(np.uint8), kernel)>0 for t in mask])
                    expected = np.stack([(mask & truth).sum((1, 2)), mask.sum((1, 2)), truth.sum((1, 2)),
                                         (mask & near_gt).sum((1, 2)), (truth & near_pred).sum((1, 2))], 1)
                    np.testing.assert_array_equal(counts[i], expected)

    def test_localization_tolerance_does_not_hide_false_positive_pixels(self):
        p = np.zeros((3, 12, 12), np.float32)
        truth = np.zeros_like(p, bool)
        truth[0, 4, :] = True
        p[0, 5, :] = .9
        p[0, 9, :] = .9
        result = structure_count_metrics(threshold_counts(p, truth, [.5])[0])
        self.assertEqual(result['exact']['channels'][0]['f1'], 0.)
        self.assertEqual(result['tolerant']['channels'][0]['precision'], .5)
        self.assertEqual(result['tolerant']['channels'][0]['recall'], 1.)
        self.assertAlmostEqual(result['tolerant']['channels'][0]['f1'], 2/3)
        self.assertIsNone(result['tolerant']['channels'][1]['f1'])

    def test_invalid_inputs_fail_explicitly(self):
        p = np.zeros((3, 5, 5), np.float32)
        for value in (np.nan, np.inf, -.1, 1.1):
            bad = p.copy(); bad[0, 0, 0] = value
            with self.assertRaises(ValueError): ridge_scores(bad)
        for grid in ([], [0], [.5, .5], [.9, .5], [np.nan], [1.1]):
            with self.assertRaises(ValueError): threshold_counts(p, p>0, grid)
        for radius in (-1, 1.5, True):
            with self.assertRaises(ValueError): ridge_scores(p, radius)
        with self.assertRaises(ValueError): structure_masks(p, [0, .5, .5])


if __name__ == '__main__':
    unittest.main()
