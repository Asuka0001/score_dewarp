import unittest

import numpy as np

from scoredewarp.core.structures import spacing_classes


class SpacingClassesTest(unittest.TestCase):
    def test_perspective_gradient_is_one_size(self):
        # one printed size, spacing shrinking towards the far edge of the photo
        sp = [11.1, 11.0, 11.0, 10.7, 10.4, 10.1, 10.3, 10.3, 10.1, 10.0, 9.7]
        self.assertEqual(set(spacing_classes(sp).tolist()), {0})

    def test_violin_above_piano_is_two_sizes(self):
        # IMG_3684-like: violin staff ~0.56 of the piano staves
        sp = [6.7, 11.8, 12.2, 6.9, 12.1, 11.9, 7.0, 11.8, 11.5]
        lab = spacing_classes(sp)
        small = np.array(sp) < 9
        self.assertEqual(len(set(lab[small])), 1)
        self.assertEqual(len(set(lab[~small])), 1)
        self.assertNotEqual(lab[small][0], lab[~small][0])

    def test_short_inputs(self):
        self.assertEqual(spacing_classes([]).tolist(), [])
        self.assertEqual(spacing_classes([12.0]).tolist(), [0])


if __name__ == "__main__":
    unittest.main()
