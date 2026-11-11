"""Structure-aware evaluation for score dewarping.

Given GT annotations (flat page + where they landed in the photo) and a
method's forward map photo -> output, every GT keypoint is pushed into the
output image and measured there. Metrics are scale-free (normalised by staff
spacing / glyph height of the *output*), so methods with different output
resolutions are comparable.

Generic
  ad_pct            aligned distortion: RMS residual after a best similarity
                    fit flat -> output, in % of page width (↓)
  coverage          fraction of GT keypoints that are inside the output (↑)
五线谱 (staff)
  staff_straight    std of y along each staff line / spacing (↓)
  staff_spacing_cv  CV of local inter-line spacing over the page (↓)
  barline_tilt_deg  |angle from vertical| of barlines (↓)
简谱 (jianpu)
  jp_baseline       std of digit bottom y along each row / digit height (↓)
  jp_height_cv      CV of digit heights over the page (↓)
  jp_tilt_deg       |angle from vertical| of digit axes (↓)
  jp_dot_err        octave-dot offset error relative to its digit / height (↓)
  jp_lyric_err      lyric-to-note horizontal offset error / height (↓)
  jp_rowgap_cv      CV of (output / flat) row-gap ratio (↓)
"""
from __future__ import annotations

import numpy as np


def _similarity_fit(A, B):
    """Umeyama: find s,R,t minimising |s R A + t - B|."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / max((A0 ** 2).sum(), 1e-12)
    t = mb - s * R @ ma
    return s, R, t


def _ang_from_vertical(top, bot):
    d = top - bot
    return np.degrees(np.abs(np.arctan2(d[:, 0], -d[:, 1])))


def evaluate(gt: dict, fwd) -> dict:
    flat_ann, photo_ann = gt["flat_ann"], gt["photo_ann"]
    page_w = gt["meta"]["size"][0]
    out = []
    for a in photo_ann:
        out.append(fwd(np.asarray(a["pts"], float)))
    flat = [np.asarray(a["pts"], float) for a in flat_ann]
    kinds = [a["kind"] for a in flat_ann]
    m = {}

    # ---- generic ------------------------------------------------------------
    keyk = {"staff_line", "jp_char", "lyric_char", "notehead", "barline", "text_char"}
    F, O = [], []
    for k, f, o in zip(kinds, flat, out):
        if k in keyk:
            F.append(f)
            O.append(o)
    F, O = np.concatenate(F), np.concatenate(O)
    ok = np.isfinite(O).all(1)
    m["coverage"] = float(ok.mean())
    if ok.sum() > 10:
        s, R, t = _similarity_fit(F[ok], O[ok])
        res = O[ok] - (s * F[ok] @ R.T + t)
        m["ad_pct"] = float(np.sqrt((res ** 2).sum(1).mean()) / (s * page_w) * 100)

    # ---- staff ------------------------------------------------------------------
    staff = {}
    for a, o in zip(flat_ann, out):
        if a["kind"] == "staff_line":
            staff.setdefault(a["staff"], {})[a["line"]] = o
    if staff:
        st_vals, sp_vals = [], []
        for sid, lines in staff.items():
            if len(lines) < 5:
                continue
            L = np.stack([lines[i] for i in range(5)])        # (5, n, 2)
            dy = np.diff(L[:, :, 1], axis=0)                   # (4, n)
            sp = np.nanmedian(dy)
            if not np.isfinite(sp) or sp <= 0:
                continue
            for i in range(5):
                y = L[i, :, 1]
                y = y[np.isfinite(y)]
                if len(y) > 5:
                    st_vals.append(np.std(y) / sp)
            sp_vals.append(dy[np.isfinite(dy)])
        if st_vals:
            m["staff_straight"] = float(np.mean(st_vals))
        if sp_vals:
            allsp = np.concatenate(sp_vals)
            m["staff_spacing_cv"] = float(np.std(allsp) / np.mean(allsp))
    bl = [(o[0], o[1]) for a, o in zip(flat_ann, out) if a["kind"] == "barline"]
    if bl:
        t = np.array([b[0] for b in bl])
        b = np.array([b[1] for b in bl])
        g = np.isfinite(t).all(1) & np.isfinite(b).all(1)
        if g.any():
            m["barline_tilt_deg"] = float(np.mean(_ang_from_vertical(t[g], b[g])))

    # ---- jianpu ---------------------------------------------------------------
    jp_idx = [i for i, k in enumerate(kinds) if k == "jp_char"]
    if jp_idx:
        top = np.array([out[i][4] for i in jp_idx])
        bot = np.array([out[i][5] for i in jp_idx])
        ctr = np.array([out[i][6] for i in jp_idx])
        ftop = np.array([flat[i][4] for i in jp_idx])
        fbot = np.array([flat[i][5] for i in jp_idx])
        h = np.linalg.norm(top - bot, axis=1)
        fh = np.linalg.norm(ftop - fbot, axis=1)
        g = np.isfinite(h)
        hmed = np.nanmedian(h)
        m["jp_height_cv"] = float(np.std(h[g]) / np.mean(h[g]))
        m["jp_tilt_deg"] = float(np.mean(_ang_from_vertical(top[g], bot[g])))
        rows = {}
        for j, i in enumerate(jp_idx):
            rows.setdefault(flat_ann[i]["row"], []).append(j)
        bs, rb, rbf = [], [], []
        for r, js in rows.items():
            y = bot[js, 1]
            y = y[np.isfinite(y)]
            if len(y) > 3:
                bs.append(np.std(y) / hmed)
                rb.append(np.median(y))
                rbf.append(np.median(fbot[js, 1]))
        if bs:
            m["jp_baseline"] = float(np.mean(bs))
        if len(rb) >= 3:
            o = np.argsort(rbf)
            ratio = np.diff(np.array(rb)[o]) / np.diff(np.array(rbf)[o])
            m["jp_rowgap_cv"] = float(np.std(ratio) / abs(np.mean(ratio)))
        pos = {i: j for j, i in enumerate(jp_idx)}
        de = []
        for a, o, f in zip(flat_ann, out, flat):
            if a["kind"] == "jp_dot" and a["char_idx"] in pos:
                j = pos[a["char_idx"]]
                d_o = (o[0] - ctr[j]) / h[j]
                d_f = (f[0] - flat[a["char_idx"]][6]) / fh[j]
                e = np.linalg.norm(d_o - d_f)
                if np.isfinite(e):
                    de.append(e)
        if de:
            m["jp_dot_err"] = float(np.mean(de))
        le = []
        for a, o, f in zip(flat_ann, out, flat):
            if a["kind"] == "lyric_char" and isinstance(a.get("note_ref"), int) and a["note_ref"] in pos:
                j = pos[a["note_ref"]]
                e = abs((o[6][0] - ctr[j][0]) / h[j] - (f[6][0] - flat[a["note_ref"]][6][0]) / fh[j])
                if np.isfinite(e):
                    le.append(e)
        if le:
            m["jp_lyric_err"] = float(np.mean(le))
    return m
