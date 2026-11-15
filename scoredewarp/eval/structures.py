"""Detection quality against annotated staff instances (not detector self-scores)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def staff_detection_metrics(gt, fwd, structures):
    """Match detected five-line systems one-to-one to visible GT systems.

    A match needs overlapping x support and median line error <= half a staff
    space. Unannotated external data returns no scores. Cropped GT systems are
    reported separately, so detection recall cannot hide missing output areas.
    """
    if not gt["flat_ann"]:
        return {}
    groups = {}
    for a in gt["photo_ann"]:
        if a["kind"] == "staff_line":
            groups.setdefault(a["staff"], {})[a["line"]] = np.asarray(fwd(np.asarray(a["pts"], float)))
    visible = []
    for lines in groups.values():
        if set(lines) != set(range(5)):
            continue
        clean = []
        for i in range(5):
            p = lines[i][np.isfinite(lines[i]).all(1)]
            p = p[np.argsort(p[:, 0])]
            clean.append(p)
        if all(len(p) >= 6 for p in clean):
            visible.append(clean)
    costs = np.full((len(visible), len(structures.staves)), 1e6)
    for i, lines in enumerate(visible):
        for j, staff in enumerate(structures.staves):
            obs = np.asarray(staff.tuples)
            x = obs[:, 0, 0]
            lo = max(p[0, 0] for p in lines)
            hi = min(p[-1, 0] for p in lines)
            keep = (x >= lo) & (x <= hi)
            if keep.sum() < 3:
                continue
            # A short accidental 5-line patch must not count as a whole system.
            if np.ptp(x[keep]) < .25 * (hi - lo):
                continue
            truth = np.stack([np.interp(x[keep], p[:, 0], p[:, 1]) for p in lines], 1)
            spacing = np.median(np.diff(truth, axis=1))
            if spacing > 0:
                costs[i, j] = np.median(np.abs(obs[keep, :, 1] - truth)) / spacing
    a, b = linear_sum_assignment(costs)
    matched = int((costs[a, b] <= .5).sum())
    n_pred, n_gt = len(structures.staves), len(groups)
    return {"staff_gt_count": n_gt, "staff_visible_gt_count": len(visible),
            "staff_pred_count": n_pred, "staff_matched": matched,
            "staff_precision": matched / n_pred if n_pred else None,
            "staff_recall": matched / n_gt if n_gt else None}
