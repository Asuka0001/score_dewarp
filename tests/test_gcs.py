import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scoredewarp.core.gcs import K_KNOTS, TABLE, GCSModel  # noqa: E402


def _model(phi, theta):
    m = GCSModel((1200, 1600))
    m.Xc, m.Yc = 0.5, 0.7
    rc = 0.6
    m.R = 1.5
    m.knots = np.linspace(-rc, rc, K_KNOTS)
    m.sg = np.linspace(-m.R, m.R, TABLE)
    m.mid = TABLE // 2
    p = np.zeros(7 + K_KNOTS)
    p[6] = phi
    p[7:] = theta
    return m, p


class GCSTests(unittest.TestCase):
    def test_finish_all_retains_candidate_that_wins_after_refinement(self):
        # Direction 0 wins the short run (10 < 20), but direction 90 wins
        # after refinement (1 < 8). No image/ground truth enters selection.
        def solve(fun, x, **kw):
            p = x.copy()
            short = p[0] == 0
            p[0] += 1
            cost = (10 if p[6] == 0 else 20) if short else (8 if p[6] == 0 else 1)
            return SimpleNamespace(x=p, cost=cost, fun=np.zeros(2))

        for finish_all, expected in ((False, 0.0), (True, np.pi / 2)):
            m = GCSModel((1200, 1600))
            m.p0 = np.zeros(7)
            m.obs, m.w = np.zeros((1, 2)), np.ones(1)
            m.residuals = lambda p: np.zeros(2)
            m.sparsity = lambda: None
            with patch("scoredewarp.core.gcs.least_squares", side_effect=solve):
                m.fit(finish_all=finish_all)
            self.assertEqual(m.p[6], expected)
            self.assertEqual(len(m.fit_candidates), 2)

    def test_surface_is_isometric_for_any_ruling_direction(self):
        rng = np.random.default_rng(0)
        for phi in (0.0, 0.4, np.pi / 2, 2.0):
            m, p = _model(phi, rng.normal(0, 0.4, K_KNOTS))
            for direction in ((1, 0), (0, 1), (0.6, 0.8)):
                t = np.linspace(-0.3, 0.3, 400)
                X = 0.5 + t * direction[0]
                Y = 0.7 + t * direction[1]
                P = m.surface(p, X, Y)
                # paper does not stretch: each small step keeps its flat length
                steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
                flat = np.hypot(np.diff(X), np.diff(Y))
                assert np.allclose(steps, flat, rtol=2e-3, atol=1e-6)


    def test_flat_profile_is_identity(self):
        m, p = _model(0.7, np.zeros(K_KNOTS))
        X, Y = np.random.default_rng(1).uniform(0, 1, (2, 50))
        P = m.surface(p, X, Y)
        assert np.allclose(P[:, 0], X) and np.allclose(P[:, 1], Y) and np.allclose(P[:, 2], 0)


    def test_bench_alignment_removes_anisotropic_scale(self):
        from bench import _align
        rng = np.random.default_rng(2)
        target = rng.uniform(0, 1, (200, 2))
        th = 0.1
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pred = (target * [1.3, 0.8]) @ R.T + [5, -2]
        assert np.abs(_align(pred, target, True) - target).max() < 1e-4
        # a similarity alignment cannot remove the per-axis scale
        assert np.abs(_align(pred, target, False) - target).max() > 1e-2


if __name__ == "__main__":
    unittest.main()
