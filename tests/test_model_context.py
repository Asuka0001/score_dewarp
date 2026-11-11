from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.learn.infer import load
from scoredewarp.learn.model import ScoreDewarpNet, build_model, load_init_weights


class ContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(2, 3, 64, 96)

    def test_zero_gate_is_identity_and_plain_weights_load(self):
        plain = ScoreDewarpNet().eval()
        ctx = ScoreDewarpNet(context="attn").eval()
        missing = load_init_weights(ctx, plain.state_dict())
        self.assertTrue(missing and all(k.startswith("context.") for k in missing))
        with torch.no_grad():
            a, b = plain(self.x), ctx(self.x)
        for key in ("fwd", "page", "kind"):
            torch.testing.assert_close(a[key], b[key])

    def test_init_rejects_non_context_mismatch(self):
        net = ScoreDewarpNet(context="attn")
        state = {k: v for k, v in ScoreDewarpNet().state_dict().items() if not k.startswith("head_fwd")}
        with self.assertRaises(RuntimeError):
            load_init_weights(net, state)

    def test_gate_receives_gradient_and_other_sizes_run(self):
        net = ScoreDewarpNet(context="attn")
        net(self.x)["fwd"].mean().backward()
        self.assertGreater(net.context.gate.grad.abs().sum().item(), 0)
        with torch.no_grad():
            out = net.eval()(torch.rand(1, 3, 96, 64))
        self.assertEqual(tuple(out["fwd"].shape), (1, 2, 96, 64))

    def test_checkpoint_arch_roundtrip_and_legacy_default(self):
        net = ScoreDewarpNet(context="attn", context_layers=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            torch.save({"model": net.state_dict(), "arch": net.arch, "size": (96, 64)}, path)
            loaded, size = load(path)
        self.assertEqual(loaded.arch["context"], "attn")
        self.assertEqual(len(loaded.context.encoder.layers), 1)
        self.assertEqual(tuple(size), (96, 64))
        self.assertIsNone(build_model({"model": {}}).context)


    def test_coord_input_init_from_plain_is_identity(self):
        plain = ScoreDewarpNet().eval()
        net = ScoreDewarpNet(context="attn", coord_input=True).eval()
        load_init_weights(net, plain.state_dict())
        self.assertEqual(net.enc[0][0].weight.shape[1], 5)
        with torch.no_grad():
            torch.testing.assert_close(plain(self.x)["fwd"], net(self.x)["fwd"])
            net.enc[0][0].weight[:, 4].fill_(1.)   # y channel now matters: output must depend on position
            self.assertFalse(torch.allclose(plain(self.x)["fwd"], net(self.x)["fwd"]))
        self.assertTrue(build_model({"arch": net.arch}).coord_input)


if __name__ == "__main__":
    unittest.main()
