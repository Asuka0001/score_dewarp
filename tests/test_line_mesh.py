import unittest

import numpy as np
import torch
import torch.nn.functional as F

from scoredewarp.core.line_mesh import fit_line_mesh
from scoredewarp.core.structures import Staff, Structures
from scoredewarp.eval.geometry import map_points
from scoredewarp.learn.uvdoc import backward_map
from scoredewarp.core.transforms import BackwardMap, mapping_quality


def page(curved=True, bad=False):
    # Independent analytic warp: equal-spaced flat lines, smoothly varying
    # displacement AND local vertical scale. Test off-grid samples as well.
    w, h = 640, 800
    xs = np.linspace(45, 595, 90)
    staves, truth = [], []
    for top in (110, 270, 430, 590):
        flat = np.stack(np.broadcast_arrays(xs[:, None], top + np.arange(5)[None] * 12), -1)
        observed = flat.copy()
        if curved:
            observed[..., 1] += 13 * np.sin(xs[:, None] / w * np.pi * 2) * np.sin(flat[..., 1] / (h - 1) * np.pi)
        if bad:
            observed[4, 2, 1] = observed[4, 1, 1] - 3
        staves.append(Staff([], list(observed), 12))
        truth.append(flat)
    return Structures((h, w), 1, 12, [], staves, [], []), truth


class LineMeshTests(unittest.TestCase):
    def test_analytic_bending_recovery_preserves_spacing_and_full_canvas(self):
        structures, truth = page()
        bm, info = fit_line_mesh(structures)
        self.assertTrue(info['accepted'], info)
        self.assertLess(np.median(info['straight_after']), .15 * np.median(info['straight_before']))
        for staff in structures.staves:
            points = np.array(staff.tuples)
            mapped, valid = map_points(bm, points)
            self.assertTrue(valid.all())
            self.assertLess(np.std(np.diff(mapped[..., 1], axis=1)) / 12, .08)
            np.testing.assert_allclose(mapped[..., 0], points[..., 0], atol=1e-4)
        corners = np.array([[0, 0], [639, 0], [0, 799], [639, 799], [320, 0], [320, 799]])
        np.testing.assert_allclose(bm.sample(corners), corners, atol=1e-4)
        self.assertGreaterEqual(info['vertical_scale_min'], .5)
        self.assertLessEqual(info['vertical_scale_max'], 2)
        self.assertEqual(info['sampled_inverse']['nonpositive_jacobian_fraction'], 0)

    def test_unobserved_symbol_points_recover_analytic_geometry(self):
        structures, _ = page()
        bm, info = fit_line_mesh(structures)
        # These points are never included in the line fit: check the blank
        # bands and note locations between staff systems against actual GT.
        x, y = np.meshgrid(np.linspace(70, 570, 13), np.linspace(140, 650, 15))
        flat = np.stack([x.ravel(), y.ravel()], 1)
        observed = flat.copy()
        observed[:, 1] += 13 * np.sin(flat[:, 0] / 640 * 2 * np.pi) * np.sin(flat[:, 1] / 799 * np.pi)
        recovered, valid = map_points(bm, observed)
        self.assertTrue(valid.all())
        error = np.linalg.norm(recovered - flat, axis=1)
        self.assertLess(np.quantile(error, .95), .5)
        self.assertLess(np.mean(error), .1 * np.mean(np.linalg.norm(observed - flat, axis=1)))

    def test_flat_input_is_not_warped(self):
        structures, _ = page(curved=False)
        bm, info = fit_line_mesh(structures)
        self.assertIsNone(bm)
        self.assertEqual(info['reason'], 'no safe structural improvement')

    def test_misordered_and_missing_evidence_are_rejected(self):
        structures, _ = page(bad=True)
        bm, info = fit_line_mesh(structures)
        self.assertIsNone(bm)
        self.assertEqual(info['rejected_systems'], 4)
        structures.staves = []
        self.assertIsNone(fit_line_mesh(structures)[0])

    def test_large_displacements_are_damped_to_remain_monotone(self):
        structures, _ = page()
        for staff in structures.staves:
            for p in staff.tuples:
                p[:, 1] += 100 * np.sin(p[:, 0] / 640 * 2 * np.pi)
        bm, info = fit_line_mesh(structures)
        if bm is not None:
            self.assertLess(info['damping'], 1)
            self.assertGreaterEqual(info['vertical_scale_min'], .5)
            self.assertLessEqual(info['vertical_shear_max'], .5)
        else:
            self.assertFalse(info['accepted'])


class UVDocAdapterTests(unittest.TestCase):
    def test_identity_endpoints_on_nondivisible_canvas(self):
        y, x = torch.meshgrid(torch.linspace(-1, 1, 7), torch.linspace(-1, 1, 5), indexing='ij')
        mesh = torch.stack([x, y])[None]
        bm = backward_map(mesh, (641, 799), (641, 799))
        points = np.array([[0, 0], [640, 798], [333, 455]], float)
        np.testing.assert_allclose(bm.sample(points), points, atol=1e-4)

    def test_nonlinear_mesh_matches_official_align_corners_sampling(self):
        y, x = torch.meshgrid(torch.linspace(-1, 1, 9), torch.linspace(-1, 1, 7), indexing='ij')
        mesh = torch.stack([x + .04 * (1 - x**2) * y, y + .06 * torch.sin(x * 2)])[None]
        # At stride 1 compare the adapter against the official continuous map.
        bm = backward_map(mesh, (321, 400), (257, 345), stride=1)
        official = F.interpolate(mesh, size=(345, 257), mode='bilinear', align_corners=True)[0].permute(1, 2, 0).numpy()
        expected = (official + 1) * np.array([320, 399]) / 2
        np.testing.assert_allclose(bm.grid[:345, :257], expected, atol=6e-5)


class MapQualityTests(unittest.TestCase):
    def test_second_triangle_fold_is_not_hidden_by_positive_top_left(self):
        grid = np.array([[[0, 0], [4, 0]], [[0, 4], [1, 1]]], float)
        quality = mapping_quality(BackwardMap(grid, 4, (5, 5)), (8, 8))
        self.assertEqual(quality['folded_cell_fraction'], 1.)
        self.assertFalse(quality['safe_local_orientation'])

    def test_identity_and_empty_maps(self):
        bm = BackwardMap.from_homography(np.eye(3), (33, 45))
        quality = mapping_quality(bm, (33, 45))
        self.assertTrue(quality['safe_local_orientation'])
        self.assertAlmostEqual(quality['anisotropy_p99'], 1)
        bm.grid[:] = np.nan
        self.assertFalse(mapping_quality(bm, (33, 45))['safe_local_orientation'])


if __name__ == '__main__':
    unittest.main()
