"""Multi-task score dewarping network (v0 skeleton).

One shared U-Net, four heads:
  fwd     (2)  dense photo -> flat page coordinates (+ page mask logit, 1)
  seg     (5)  off-page / paper / staff region / jianpu region / text-lyrics
  struct  (3)  staff-line, jianpu-baseline, vertical-line heatmaps
  kind    (3)  page type: staff / jianpu / mixed  (global pooled bottleneck)

Losses (see `losses`):
  * L1 on coordinates + gradient (local-distortion) loss, masked to the page
  * type-specific *structure consistency* losses, which encode the same
    priors as the classical optimiser but differentiably:
      - every staff line / jianpu baseline instance must have a constant
        predicted flat-Y  (variance over the instance's pixels)
      - every barline/stem instance must have constant predicted flat-X
  * CE / BCE for seg, struct, kind.
The struct heads can also feed the classical optimiser (replacing the
handcrafted detectors) — see README "roadmap".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class ScoreDewarpNet(nn.Module):
    def __init__(self, ch=(32, 64, 128, 256, 256)):
        super().__init__()
        self.enc = nn.ModuleList()
        ci = 3
        for c in ch:
            self.enc.append(_block(ci, c))
            ci = c
        self.dec = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.dec.append(_block(ch[i] + ch[i - 1], ch[i - 1]))
        c0 = ch[0]
        self.head_fwd = nn.Conv2d(c0, 3, 1)
        self.head_seg = nn.Conv2d(c0, 5, 1)
        self.head_struct = nn.Conv2d(c0, 3, 1)
        self.head_kind = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch[-1], 64), nn.ReLU(),
                                       nn.Linear(64, 3))

    def forward(self, x):
        feats = []
        for i, b in enumerate(self.enc):
            x = b(x if i == 0 else F.max_pool2d(x, 2))
            feats.append(x)
        kind = self.head_kind(x)
        for j, b in enumerate(self.dec):
            skip = feats[-2 - j]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = b(torch.cat([x, skip], 1))
        f = self.head_fwd(x)
        return {"fwd": torch.sigmoid(f[:, :2]), "page": f[:, 2:3], "seg": self.head_seg(x),
                "struct": self.head_struct(x), "kind": kind}


def _instance_variance(val, inst):
    """Mean over instances of the variance of `val` (B,H,W) within each instance id (B,H,W)."""
    B = val.shape[0]
    losses = []
    for b in range(B):
        ids = inst[b].reshape(-1)
        v = val[b].reshape(-1)
        m = ids > 0
        if m.sum() < 2:
            continue
        ids, v = ids[m], v[m]
        n = int(ids.max()) + 1
        cnt = torch.zeros(n, device=v.device).index_add_(0, ids, torch.ones_like(v))
        s1 = torch.zeros(n, device=v.device).index_add_(0, ids, v)
        s2 = torch.zeros(n, device=v.device).index_add_(0, ids, v * v)
        ok = cnt > 3
        var = s2[ok] / cnt[ok] - (s1[ok] / cnt[ok]) ** 2
        losses.append(var.clamp_min(0).mean())
    return torch.stack(losses).mean() if losses else val.sum() * 0


def losses(out, batch, w=None):
    w = w or {"fwd": 1.0, "grad": 0.5, "page": 0.2, "seg": 0.2, "struct": 0.2, "kind": 0.1, "cons": 50.0}
    page = batch["page"]
    pf, gf = out["fwd"], batch["fwd"]
    L = {}
    L["fwd"] = (torch.abs(pf - gf) * page).sum() / page.sum().clamp_min(1) / 2
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    mx, my = page[..., :, 1:] * page[..., :, :-1], page[..., 1:, :] * page[..., :-1, :]
    L["grad"] = ((torch.abs(dx(pf) - dx(gf)) * mx).sum() / mx.sum().clamp_min(1) +
                 (torch.abs(dy(pf) - dy(gf)) * my).sum() / my.sum().clamp_min(1)) * 10
    L["page"] = F.binary_cross_entropy_with_logits(out["page"], page)
    def supervised_mean(value, head):
        # External datasets may lack score-specific labels. Missing labels must
        # not become negative examples. Works for mixed-source minibatches too.
        per_sample = value.reshape(value.shape[0], -1).mean(dim=1)
        valid = batch.get(head + "_valid", torch.ones_like(per_sample)).to(per_sample)
        return (per_sample * valid).sum() / valid.sum().clamp_min(1)

    L["seg"] = supervised_mean(F.cross_entropy(out["seg"], batch["seg"], reduction="none"), "seg")
    L["struct"] = supervised_mean(F.binary_cross_entropy_with_logits(
        out["struct"], batch["struct"], reduction="none",
        pos_weight=torch.tensor(20.0, device=pf.device)), "struct")
    L["kind"] = supervised_mean(F.cross_entropy(out["kind"], batch["kind"], reduction="none"), "kind")
    # structure consistency: horizontal structures -> constant flat Y; vertical -> constant flat X
    L["cons"] = _instance_variance(pf[:, 1], batch["inst_h"]) + _instance_variance(pf[:, 0], batch["inst_v"])
    total = sum(w[k] * v for k, v in L.items())
    return total, {k: float(v.detach()) for k, v in L.items()}
