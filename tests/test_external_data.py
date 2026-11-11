import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from scoredewarp.learn.dataset import build_targets
from scoredewarp.learn.model import losses

spec = importlib.util.spec_from_file_location("prepare", Path(__file__).resolve().parents[1] / "scripts/prepare_datasets.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

download_spec = importlib.util.spec_from_file_location(
    "download", Path(__file__).resolve().parents[1] / "scripts/download_datasets.py")
download = importlib.util.module_from_spec(download_spec)
download_spec.loader.exec_module(download)


class ExternalDataTests(unittest.TestCase):
    def test_doc3d_children_share_parent_split(self):
        parent = prepare.doc3d_group("179_7-texture-AbC0001")
        self.assertEqual(parent, "179_7")
        for stem in ("1_179_7-texture-XyZ0001", "2_179_7-other-Def0001"):
            self.assertEqual(prepare.doc3d_group(stem), parent)
            self.assertEqual(prepare.split_for(prepare.doc3d_group(stem), 7), prepare.split_for(parent, 7))
        with self.assertRaises(ValueError):
            prepare.doc3d_group("unknown-texture-AbC0001")

    def test_gated_download_explains_access_requirement_without_token(self):
        with patch.object(download.subprocess, "run", return_value=SimpleNamespace(stdout="403", returncode=22)):
            with self.assertRaises(SystemExit) as error:
                download.curl(download.HF_REPO + "/resolve/main/doc3d/img_8.zip", Path("unused"), token="secret")
        self.assertIn("request dataset access", str(error.exception))
        self.assertNotIn("secret", str(error.exception))

    def test_uv_axes_validity_and_loader_roundtrip(self):
        # Identity page with deliberately different x/y coordinates catches
        # channel swaps and a missing Blender-V flip.
        uv = np.zeros((16, 16, 3), np.float32)
        uv[2:14, 2:14] = [1, 0.75, 0.125]
        mapping = prepare.doc3d_uv_to_map(uv)
        np.testing.assert_allclose(mapping[5, 6], [56, 112])
        self.assertTrue(np.isnan(mapping[0, 0]).all())
        with tempfile.TemporaryDirectory() as td:
            prepare.save_sample(Path(td), np.zeros((16, 16, 3), np.uint8), mapping,
                                {"size": [448, 448], "kind": "unknown",
                                 "supervision": {"seg": False, "struct": False, "kind": False}})
            d = build_targets(td, (16, 16))
            np.testing.assert_allclose(d["fwd"][:, 5, 6], [0.125, 0.25])
            self.assertEqual(d["page"][0, 0, 0], 0)
            self.assertEqual(d["kind_valid"], 0)

    def test_missing_labels_have_zero_auxiliary_gradient(self):
        out = {"fwd": torch.rand(2, 2, 8, 8, requires_grad=True),
               "page": torch.randn(2, 1, 8, 8, requires_grad=True),
               "seg": torch.randn(2, 5, 8, 8, requires_grad=True),
               "struct": torch.randn(2, 3, 8, 8, requires_grad=True),
               "kind": torch.randn(2, 3, requires_grad=True)}
        batch = {"fwd": torch.zeros(2, 2, 8, 8), "page": torch.ones(2, 1, 8, 8),
                 "seg": torch.zeros(2, 8, 8, dtype=torch.long),
                 "struct": torch.zeros(2, 3, 8, 8), "kind": torch.zeros(2, dtype=torch.long),
                 "inst_h": torch.zeros(2, 8, 8, dtype=torch.long),
                 "inst_v": torch.zeros(2, 8, 8, dtype=torch.long),
                 "seg_valid": torch.tensor([0., 1.]), "struct_valid": torch.tensor([0., 1.]),
                 "kind_valid": torch.tensor([0., 1.])}
        loss, _ = losses(out, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for head in ("seg", "struct", "kind"):
            self.assertEqual(out[head].grad[0].abs().sum().item(), 0)
            self.assertGreater(out[head].grad[1].abs().sum().item(), 0)
        self.assertGreater(out["fwd"].grad[0].abs().sum().item(), 0)
        for head in ("seg", "struct", "kind"):
            batch[head + "_valid"].zero_()
        _, parts = losses(out, batch)
        self.assertTrue(all(parts[h] == 0 for h in ("seg", "struct", "kind")))


if __name__ == "__main__":
    unittest.main()
