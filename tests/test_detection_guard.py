import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from test_detection_route import guard_choice, map_pass


class DetectionRouteGuardTest(unittest.TestCase):
    def row(self, folded=0., anisotropy=2., checked=10):
        return {"notes": {"mapping_quality": {"checked_cells": checked,
                "folded_cell_fraction": folded, "anisotropy_p99": anisotropy}}}

    def test_fallback_uses_map_checks_without_ground_truth(self):
        base, learned = self.row(), self.row(folded=.1)
        self.assertEqual(guard_choice(base, learned), ("classical", False))
        # Even a favourable offline accuracy metric must not override a fold.
        base["ad_c"], learned["ad_c"] = 100., 0.
        self.assertEqual(guard_choice(base, learned), ("classical", False))

    def test_both_invalid_remain_explicitly_unresolved(self):
        self.assertEqual(guard_choice(self.row(checked=0), self.row(anisotropy=25.)),
                         ("homr", True))
        self.assertFalse(map_pass({}))
