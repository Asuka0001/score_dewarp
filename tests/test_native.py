import unittest

import numpy as np

from scoredewarp.core.structures import Blob, Fragment, Staff, Structures, TextRow, VSeg, scale_structures
from scoredewarp.core.transforms import BackwardMap, native_scale


class NativeResolutionTest(unittest.TestCase):
    def test_native_scale_of_uniform_map(self):
        # output pixel -> 2 photo pixels in each direction
        bm = BackwardMap.from_homography(np.diag([0.5, 0.5, 1.0]), (400, 300))
        self.assertAlmostEqual(native_scale(bm, (800, 600)), 2.0, places=3)

    def test_rescale_keeps_geometry(self):
        H = np.array([[0.6, 0.05, 10], [0.02, 0.55, 5], [1e-5, 2e-5, 1.0]])
        bm = BackwardMap.from_homography(H, (400, 300))
        big = bm.rescale(2.0)
        self.assertEqual(big.out_size, (800, 600))
        q = np.array([[10.0, 20.0], [200.0, 150.0], [390.0, 290.0]])
        np.testing.assert_allclose(big.sample(q * 2), bm.sample(q), atol=0.05)
        self.assertAlmostEqual(native_scale(big, (2000, 2000)), native_scale(bm, (2000, 2000)) / 2, places=2)

    def test_scale_structures(self):
        st = Staff(frag_ids=[0], tuples=[np.array([[10.0, 20 + 5 * i] for i in range(5)])], spacing=5.0,
                   top_poly=np.array([0.001, 0.1, 20.0]), bot_poly=np.array([0.0, 40.0]), x_range=(10, 90))
        S = Structures((100, 120), 1.5, 5.0, [Fragment(np.array([[0.0, 1.0], [10.0, 2.0]]), 0, 0)], [st],
                       [TextRow([Blob(1, 2, 3, 4, "5")], "jianpu", 2.0)], [VSeg(np.array([[5.0, 5.0], [5.0, 15.0]]))],
                       regions=[("staff", np.array([[0.0, 0.0], [10.0, 10.0]]))])
        T = scale_structures(S, 2.0)
        self.assertEqual(T.shape, (200, 240))
        self.assertEqual(T.spacing_est, 10.0)
        np.testing.assert_allclose(T.fragments[0].pts, S.fragments[0].pts * 2)
        x = 50.0
        self.assertAlmostEqual(np.polyval(T.staves[0].top_poly, 2 * x), 2 * np.polyval(st.top_poly, x))
        self.assertAlmostEqual(np.polyval(T.staves[0].bot_poly, 2 * x), 2 * np.polyval(st.bot_poly, x))
        self.assertEqual((T.rows[0].blobs[0].x1, T.rows[0].height), (6, 4.0))
        self.assertEqual(S.rows[0].blobs[0].x1, 3)   # original untouched


if __name__ == "__main__":
    unittest.main()
