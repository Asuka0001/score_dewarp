import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scoredewarp.core.field import _guard_field, _vertical_support


class FieldGuardTests(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(np.arange(4) * 48., np.arange(5) * 48.)
        self.identity = np.stack([x, y], -1)

    def test_translation_and_mild_shear_are_unchanged(self):
        field = self.identity @ np.array([[1., .1], [0., 1.]]) + [20, -10]
        accepted, step = _guard_field(field, self.identity, 48)
        self.assertEqual(step, 1.)
        np.testing.assert_allclose(accepted, field)

    def test_reflection_and_excessive_stretch_are_damped(self):
        for scale in ([-1, 1], [20, 1], [.01, 1]):
            field, step = _guard_field(self.identity * scale, self.identity, 48)
            self.assertGreater(step, 0.)
            self.assertLess(step, 1.)
            self.assertTrue((np.diff(field[..., 0], axis=1) >= .2 * 48).all())
            self.assertTrue((np.diff(field[..., 0], axis=1) <= 5 * 48).all())

    def test_corner_fold_is_detected_even_when_cell_centre_is_positive(self):
        identity = self.identity[:2, :2]
        field = identity.copy()
        field[1, 1] = [-12, 48]
        _, step = _guard_field(field, identity, 48)
        self.assertLess(step, 1.)

    def test_nonfinite_field_reverts_to_identity(self):
        field = self.identity.copy()
        field[2, 2] = np.nan
        accepted, step = _guard_field(field, self.identity.astype(int), 48)
        self.assertEqual(step, 0.)
        np.testing.assert_array_equal(accepted, self.identity)
        self.assertEqual(accepted.dtype.kind, "f")

    def test_staff_crossing_support_preserves_barlines_without_angle_filter(self):
        st = SimpleNamespace(spacing=10., tuples=[np.array([[x, y] for y in range(20, 61, 10)])
                                               for x in (0., 100.)])
        self.assertEqual(_vertical_support(np.array([[40., 20.], [50., 60.]]), [st]), 1.)
        self.assertLess(_vertical_support(np.array([[40., 30.], [40., 40.]]), [st]), 1.)
        self.assertEqual(_vertical_support(np.array([[40., 30.], [40., 40.]]), []), 1.)

    def test_rejected_field_preserves_image_map_and_content(self):
        from scoredewarp.core.pipeline import Config, _finish
        from scoredewarp.core.structures import Structures
        from scoredewarp.core.transforms import BackwardMap

        image = np.full((48, 48, 3), 255, np.uint8)
        evidence = Structures((48, 48), 1., 10., [], [], [], [])
        bmap = BackwardMap.from_homography(np.eye(3), (48, 48))
        points = np.array([[5., 5.], [40., 40.]])
        with patch("scoredewarp.core.pipeline._detect", return_value=(image, image, evidence)), \
                patch("scoredewarp.core.pipeline.fit_field", return_value=SimpleNamespace(field_step=0.)) as fit:
            result = _finish(image, image, bmap, Config(field_iters=3), {}, {}, 0., False,
                             keep_pts=points)
        self.assertEqual(fit.call_count, 1)
        self.assertIs(result.bmap, bmap)
        self.assertIs(result.image, image)
        np.testing.assert_array_equal(result.content, points)
        self.assertEqual(result.stages["field_steps"], [0.])
