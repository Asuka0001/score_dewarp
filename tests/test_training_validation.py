import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("train", Path(__file__).resolve().parents[1] / "scripts/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class ValidationTests(unittest.TestCase):
    def test_epe_counts_all_gt_pixels_even_when_page_prediction_is_empty(self):
        class FixedNet(torch.nn.Module):
            def forward(self, image):
                fwd = torch.zeros(2, 2, 8, 8)
                fwd[0, 0] = .1
                fwd[1, 0] = .3
                return {"fwd": fwd, "page": torch.full((2, 1, 8, 8), -100.),
                        "seg": torch.zeros(2, 5, 8, 8), "struct": torch.zeros(2, 3, 8, 8),
                        "kind": torch.zeros(2, 3)}
        page = torch.ones(2, 1, 8, 8)
        page[0, :, 1:] = 0  # unequal page areas must still receive equal per-page weight
        batch = {"image": torch.zeros(2, 3, 8, 8), "page": page, "fwd": torch.zeros(2, 2, 8, 8),
                 "seg": torch.zeros(2, 8, 8, dtype=torch.long), "struct": torch.zeros(2, 3, 8, 8),
                 "kind": torch.zeros(2, dtype=torch.long), "inst_h": torch.zeros(2, 8, 8, dtype=torch.long),
                 "inst_v": torch.zeros(2, 8, 8, dtype=torch.long),
                 **{head + "_valid": torch.ones(2) for head in ("seg", "struct", "kind")}}
        metrics = train.validate(FixedNet(), [batch], "cpu", True)
        self.assertAlmostEqual(metrics["coord_epe_pct"], 20., places=4)
        self.assertEqual(metrics["page_iou"], 0.)


if __name__ == "__main__":
    unittest.main()
