import unittest
from types import SimpleNamespace

import numpy as np

from scoredewarp.synth.distort import projection_diagnostics


class ProjectionTests(unittest.TestCase):
    def test_identity_and_sheared_projection_pass(self):
        for shear in (0., .4):
            warp = SimpleNamespace(Wf=200, Hf=300,
                                   forward=lambda p: np.asarray(p).reshape(-1, 2) @ np.array([[1., shear], [0., 1.]]))
            self.assertTrue(projection_diagnostics(warp)["valid"])

    def test_fold_and_near_tangent_projection_fail(self):
        def folded(p):
            p = np.asarray(p).reshape(-1, 2).copy()
            p[:, 0] = 200 * np.sin(np.pi * p[:, 0] / 200)
            return p
        warp = SimpleNamespace(Wf=200, Hf=300, forward=folded)
        self.assertFalse(projection_diagnostics(warp)["valid"])
        def tangent(p):
            p = np.asarray(p).reshape(-1, 2).copy()
            p[:, 0] = p[:, 0] ** 3 / 200 ** 2
            return p
        warp.forward = tangent
        self.assertFalse(projection_diagnostics(warp)["valid"])


if __name__ == "__main__":
    unittest.main()
