import importlib.util
from pathlib import Path
import unittest

import torch
from scoredewarp.learn.model import _instance_variance, losses

spec = importlib.util.spec_from_file_location("train", Path(__file__).resolve().parents[1] / "scripts/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class ValidationTests(unittest.TestCase):
    def test_geometry_consistency_keeps_only_in_page_instances_and_disables_aux_heads(self):
        batch = {"page": torch.tensor([[[[1., 0.], [1., 1.]]]]),
                 "inst_h": torch.ones(1, 2, 2, dtype=torch.long),
                 "inst_v": torch.full((1, 2, 2), 2, dtype=torch.long),
                 **{head + "_valid": torch.ones(1) for head in ("seg", "struct", "kind")}}
        plain = train.geometry_batch(batch)
        consistent = train.geometry_batch(batch, keep_instances=True)
        self.assertEqual(plain["inst_h"].sum(), 0)
        self.assertEqual(consistent["inst_h"].sum(), 3)
        self.assertEqual(consistent["inst_v"].sum(), 6)
        self.assertEqual(batch["inst_h"].sum(), 4)
        for head in ("seg", "struct", "kind"):
            self.assertEqual(consistent[head + "_valid"].sum(), 0)

    def test_short_instances_have_finite_zero_loss_and_long_instances_receive_gradient(self):
        val = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4).requires_grad_()
        short = torch.tensor([[[1, 1, 2, 2], [3, 3, 4, 4]]])
        empty = _instance_variance(val, short)
        self.assertEqual(empty.item(), 0.)
        loss = _instance_variance(val, torch.ones_like(short))
        loss.backward()
        self.assertGreater(loss.item(), 0.)
        self.assertTrue(torch.isfinite(val.grad).all())
        self.assertGreater(val.grad.abs().sum(), 0.)

    def test_partial_consistency_weight_override_preserves_other_losses(self):
        out = {"fwd": torch.randn(1, 2, 4, 4, requires_grad=True),
               "page": torch.zeros(1, 1, 4, 4), "seg": torch.zeros(1, 5, 4, 4),
               "struct": torch.zeros(1, 3, 4, 4), "kind": torch.zeros(1, 3)}
        batch = {"page": torch.ones(1, 1, 4, 4), "fwd": torch.zeros(1, 2, 4, 4),
                 "seg": torch.zeros(1, 4, 4, dtype=torch.long), "struct": torch.zeros(1, 3, 4, 4),
                 "kind": torch.zeros(1, dtype=torch.long),
                 "inst_h": torch.ones(1, 4, 4, dtype=torch.long),
                 "inst_v": torch.zeros(1, 4, 4, dtype=torch.long)}
        full, parts = losses(out, batch)
        plain, _ = losses(out, batch, w={"cons": 0.})
        self.assertAlmostEqual((full - plain).item(), 50. * parts["cons"], places=4)
        plain.backward()
        self.assertGreater(out["fwd"].grad.abs().sum(), 0.)

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
