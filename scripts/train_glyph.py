"""Train the small jianpu glyph classifier (digits 0-7 vs. everything else).

Crops are made the way the detector makes them (``structures.glyph_blobs``):
one binarised connected component, so the network sees exactly what the
pipeline gives it. Glyphs are drawn with SYSTEM fonts that the bench renderer
does not use (``synth.fonts`` digit fonts Arial / Times New Roman / Helvetica
and its CJK lyric fonts are excluded), so dev / test jianpu pages measure
generalisation to unseen fonts.

Classes: 0..7 (jianpu digits), 8 = other (CJK, Latin letters, 8 / 9,
punctuation, dashes, dots, broken / merged glyph pieces).

  python scripts/train_glyph.py --out models/glyph            # ~5 min on Apple Silicon (MPS)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.core.glyphnet import SIZE, prepare  # noqa: E402

SUP = "/System/Library/Fonts/Supplemental/"
SYS = "/System/Library/Fonts/"
DIGIT_FONTS = [SYS + f for f in ("Avenir.ttc", "Avenir Next.ttc", "HelveticaNeue.ttc", "Courier.ttc", "Menlo.ttc",
                                 "Monaco.ttf", "Geneva.ttf", "Optima.ttc", "Palatino.ttc", "Times.ttc", "NewYork.ttf",
                                 "SFNS.ttf", "SFCompact.ttf", "SFNSRounded.ttf", "SFNSMono.ttf",
                                 "ヒラギノ角ゴシック W3.ttc", "ヒラギノ角ゴシック W6.ttc", "ヒラギノ明朝 ProN.ttc",
                                 "ヒラギノ丸ゴ ProN W4.ttc")] + \
              [SUP + f for f in ("Georgia.ttf", "Georgia Bold.ttf", "Tahoma.ttf", "Tahoma Bold.ttf", "Futura.ttc",
                                 "GillSans.ttc", "Baskerville.ttc", "Didot.ttc", "PTSans.ttc", "PTSerif.ttc",
                                 "Charter.ttc", "Rockwell.ttc", "Microsoft Sans Serif.ttf", "DIN Alternate Bold.ttf",
                                 "Courier New.ttf", "Courier New Bold.ttf", "Comic Sans MS.ttf", "Andale Mono.ttf",
                                 "Cochin.ttc", "Hoefler Text.ttc", "Iowan Old Style.ttc", "Seravek.ttc", "Skia.ttf",
                                 "STIXTwoText.ttf", "NISC18030.ttf", "Verdana.ttf", "Verdana Bold.ttf",
                                 "Trebuchet MS.ttf", "Impact.ttf", "BigCaslon.ttf", "AmericanTypewriter.ttc")]
CJK_FONTS = [SUP + "NISC18030.ttf", SYS + "ヒラギノ明朝 ProN.ttc", SYS + "ヒラギノ角ゴシック W3.ttc",
             SYS + "ヒラギノ角ゴシック W6.ttc", SYS + "ヒラギノ丸ゴ ProN W4.ttc", SUP + "Arial Unicode.ttf"]
CJK = list("的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学"
           "么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名"
           "法间斯知世什两次使身者被高已亲其进此话常与活正感见明问力理尔点文几定本公特做外孩相西果走将月十实向声车全信重三机工物气每并"
           "别真打太新比才便夫再书部水像眼等体却加电主界门利海受听表德少克代员许稜先口由死安写性马光白或住难望教命花结乐色更拉东神记处"
           "让母父应直字场平报友关放至张认接告入笑内英军候民岁往何度山觉路带万男边风解叫任金快原吃妈变通师立象数四失满战远格士音轻目条"
           "呢病始达深完今提求清王化空业思切怎非找片罗钱紶吗语元喜曾离飞科言干流欢约各即指合反题必该论交终林请医晚制球决窢传画保读运及"
           "则房早院量苦火布品近坐产答星精视五连司巴奇管类未朋且婚台夜青北队久乎越观落尽形影红爸百令周吧识步希亚术留市半热送兴造谈容极"
           "随演收首根讲整式取照办强石古华諣拿计您装似足双妻尼转诉米称丽客南领节衣站黑刻统断福城故历惊脸选包紧争另建维绝树系伤示愿持千史"
           "谁准联妇纪基买志静阿诗独复痛消社算义竟确酒需单治卡幸兰念举仅钟怕共毛句息功官待究跟穿室易游程号居考突皮哪费倒价图具刚脑永歌响"
           "商礼细专黄块脚味灵改据般破引食仍存众注笔甚某沉血备习校默务土微娘须试怀料调广蜖苏显赛查密议底列富梦错座参八除跑亮假印设线温虽")


def _usable(paths, probe):
    ok = []
    for p in paths:
        try:
            if not os.path.exists(p):
                continue
            f = ImageFont.truetype(p, 64)
            img = Image.new("L", (256, 192), 0)
            ImageDraw.Draw(img).text((128, 128), probe, font=f, fill=255, anchor="ms")
            if np.asarray(img).max() > 127:
                ok.append(p)
        except OSError:
            pass
    return ok


DIGIT_FONTS = _usable(DIGIT_FONTS, "5")
CJK_FONTS = _usable(CJK_FONTS, "春")
LATIN = list("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz89#%&?!()[]{}<>/=+~@$")   # no l I | O o (look like 1 / 0)


def _font(path, size, rng):
    if path.endswith(".ttc"):
        try:
            return ImageFont.truetype(path, size, index=int(rng.integers(0, 3)))
        except OSError:
            pass
    return ImageFont.truetype(path, size)


def _draw(text, path, rng, size=64):
    img = Image.new("L", (size * 4, size * 3), 0)
    try:
        ImageDraw.Draw(img).text((size * 2, size * 2), text, font=_font(path, size, rng), fill=255, anchor="ms")
    except OSError:     # a few collection faces fail to render
        return np.zeros((size * 3, size * 4), np.uint8)
    return np.array(img)


def _augment(a, rng, h_target):
    """Glyph image (white on black, ~64 px) -> binarised component crop as the detector sees it."""
    ys, xs = np.nonzero(a > 127)
    if len(ys) < 10:
        return None
    a = a[max(0, ys.min() - 8):ys.max() + 9, max(0, xs.min() - 8):xs.max() + 9].astype(np.float32)
    h, w = a.shape
    # stroke weight
    k = int(rng.integers(-2, 3))
    if k > 0:
        a = cv2.dilate(a, np.ones((k + 1, k + 1), np.uint8))
    elif k < 0:
        a = cv2.erode(a, np.ones((-k + 1, -k + 1), np.uint8))
    # small rotation / shear / perspective
    ang = np.radians(rng.normal(0, 3))
    sh = rng.normal(0, 0.06)
    M = np.array([[np.cos(ang), -np.sin(ang) + sh, 0], [np.sin(ang), np.cos(ang), 0]], np.float32)
    c = np.array([w / 2, h / 2])
    M[:, 2] = c - M[:, :2] @ c
    sx = rng.uniform(0.85, 1.15)
    M[0, :] *= sx
    a = cv2.warpAffine(a, M, (int(w * 1.3) + 4, h + 4), borderValue=0)
    # downsample to the target glyph height, blur, noise, then threshold like Otsu
    ys = np.nonzero(a.max(1) > 127)[0]
    gh = max(int(np.ptp(ys)) + 1 if len(ys) else h, 1)
    f = h_target / gh
    small = cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    if rng.random() < 0.6:
        small = cv2.GaussianBlur(small, (0, 0), rng.uniform(0.3, 1.0))
    small = small * rng.uniform(0.5, 1.0) + rng.normal(0, rng.uniform(0, 18), small.shape)
    thr = rng.uniform(0.35, 0.6) * small.max()
    ink = (small > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w2, h2, _ = st[i]
    return (lab[y:y + h2, x:x + w2] == i)


def _other(rng):
    r = rng.random()
    if r < 0.45:
        return _draw(str(rng.choice(CJK)), str(rng.choice(CJK_FONTS)), rng)
    if r < 0.75:
        return _draw(str(rng.choice(LATIN)), str(rng.choice(DIGIT_FONTS)), rng)
    if r < 0.88:   # two touching glyphs: one blob, not one digit
        path = str(rng.choice(DIGIT_FONTS))
        a = _draw(str(rng.integers(0, 10)), path, rng)
        b = _draw(str(rng.choice(list("0123456789abcdefg"))), path, rng)
        xa, xb = np.nonzero(a.max(0))[0], np.nonzero(b.max(0))[0]
        if not len(xa) or not len(xb):
            return a
        shift = xa.max() - xb.min() - int(rng.integers(1, 4))
        return np.maximum(a, np.roll(b, shift, axis=1))
    # broken piece of a CJK char (half of it); a cut digit can still look like a digit
    a = _draw(str(rng.choice(CJK)), str(rng.choice(CJK_FONTS)), rng)
    ys, xs = np.nonzero(a)
    if not len(ys):
        return a
    if rng.random() < 0.5:
        a[(ys.min() + ys.max()) // 2:] = 0
    else:
        a[:, (xs.min() + xs.max()) // 2:] = 0
    return a


def make_set(n, seed):
    rng = np.random.default_rng(seed)
    X, Y = np.zeros((n, 1, SIZE, SIZE), np.float32), np.zeros(n, np.int64)
    i = 0
    while i < n:
        if rng.random() < 0.55:
            y = int(rng.integers(0, 8))
            a = _draw(str(y), str(rng.choice(DIGIT_FONTS)), rng)
        else:
            y = 8
            a = _other(rng)
        crop = _augment(a, rng, rng.uniform(9, 48))
        if crop is None or crop.shape[0] < 5:
            continue
        X[i, 0] = prepare(crop)
        Y[i] = y
        i += 1
    return X, Y


def main():
    import torch
    import torch.nn as nn
    from scoredewarp.core.glyphnet import build_torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("models/glyph"))
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / "_train_cache.npz"
    if cache.exists():
        d = np.load(cache)
        X, Y, Xv, Yv = d["X"], d["Y"], d["Xv"], d["Yv"]
    else:
        X, Y = make_set(a.n, 1)
        Xv, Yv = make_set(8000, 2)
        np.savez_compressed(cache, X=X, Y=Y, Xv=Xv, Yv=Yv)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = build_torch().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * (len(X) // 256 + 1))
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.02)
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for k in range(0, len(Xt), 256):
            idx = perm[k:k + 256]
            xb, yb = Xt[idx].to(dev), Yt[idx].to(dev)
            if torch.rand(1).item() < 0.5:   # random 1-px shifts
                xb = torch.roll(xb, shifts=(int(torch.randint(-1, 2, (1,))), int(torch.randint(-1, 2, (1,)))), dims=(2, 3))
            loss = lossf(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            pv = net(torch.from_numpy(Xv).to(dev)).argmax(1).cpu().numpy()
        dig = Yv < 8
        print(f"epoch {ep + 1}: loss {loss.item():.3f} val acc {np.mean(pv == Yv):.4f} "
              f"digit recall {np.mean(pv[dig] == Yv[dig]):.4f} other->digit {np.mean(pv[~dig] < 8):.4f}", flush=True)
    net = net.cpu().eval()
    torch.save(net.state_dict(), a.out / "_glyph_state.pt")
    torch.onnx.export(net, torch.zeros(1, 1, SIZE, SIZE), str(a.out / "glyph.onnx"), input_names=["x"],
                      output_names=["logits"], dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17,
                      dynamo=False)
    print("saved", a.out / "glyph.onnx")


if __name__ == "__main__":
    main()
