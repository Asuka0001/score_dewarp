import unittest
import cv2
import numpy as np

from scoredewarp.core.preprocess import binarize
from scoredewarp.core.structures import (Fragment, classify_digit, detect_structures,
                                         group_staves)


class StructureDetectionTests(unittest.TestCase):
    def test_interrupted_staff_is_joined_without_inventing_gap_points(self):
        fragments = []
        for left in range(30, 530, 100):
            for line in range(5):
                x = np.arange(left, left + 81, 3)
                fragments.append(Fragment(np.stack([x, np.full_like(x, 60 + 12 * line)], 1)))
        staves = group_staves(fragments, 600, 12)
        self.assertEqual(len(staves), 1)
        self.assertEqual(len(staves[0].frag_ids), 25)
        # Every exported tuple is supported by actual observed fragments.
        for t in staves[0].tuples:
            self.assertTrue(any(left <= t[0, 0] <= left + 80 for left in range(30, 530, 100)))

    def test_neighbouring_staves_do_not_merge(self):
        fragments = []
        for top in (40, 150):
            for left in range(20, 520, 100):
                for line in range(5):
                    x = np.arange(left, left + 81, 3)
                    fragments.append(Fragment(np.stack([x, np.full_like(x, top + 12 * line)], 1)))
        staves = group_staves(fragments, 600, 12)
        self.assertEqual(len(staves), 2)
        self.assertEqual({f.staff for f in fragments}, {0, 1})

    def test_low_contrast_staff_survives_dark_noteheads(self):
        norm = np.full((360, 600), 255, np.uint8)
        for top in (60, 210):
            for line in range(5):
                cv2.line(norm, (30, top + line * 10), (570, top + line * 10), 220, 2)
            for x in range(80, 550, 70):
                cv2.ellipse(norm, (x, top + 25), (7, 5), 0, 0, 360, 20, -1)
                cv2.line(norm, (x + 6, top + 25), (x + 6, top - 10), 20, 2)
        ink = binarize(norm)
        self.assertEqual(ink[60, 200], 0)  # global threshold really lost this line
        result = detect_structures(norm, ink)
        self.assertEqual(result.page_type, 'staff')
        self.assertEqual(len(result.staves), 2)
        self.assertTrue(all(len(set(result.fragments[i].line for i in s.frag_ids)) == 5
                            for s in result.staves))
        # Newly recovered observations are not silently fed into a geometry
        # optimizer whose weights were calibrated on the original detector.
        conservative = detect_structures(norm, ink, robust=False)
        self.assertEqual(len(conservative.staves), 0)

    def test_blank_and_four_lines_do_not_become_a_staff(self):
        for lines in (0, 4):
            norm = np.full((300, 600), 255, np.uint8)
            for i in range(lines):
                cv2.line(norm, (30, 70 + i * 14), (570, 70 + i * 14), 220, 2)
            result = detect_structures(norm, binarize(norm))
            self.assertEqual(len(result.staves), 0)
            self.assertEqual(result.page_type, 'unknown')

    def test_missing_font_boxes_are_not_zeros_but_round_zeros_are(self):
        box = np.zeros((40, 26), np.uint8)
        cv2.rectangle(box, (0, 0), (25, 39), 1, 3)
        self.assertIsNone(classify_digit(box)[0])
        zero = np.zeros((40, 26), np.uint8)
        cv2.ellipse(zero, (13, 20), (11, 18), 0, 0, 360, 1, 3)
        self.assertEqual(classify_digit(zero)[0], '0')


if __name__ == '__main__':
    unittest.main()
