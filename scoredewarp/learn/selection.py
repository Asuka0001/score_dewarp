"""Domain-by-domain checkpoint gates, independent of validation set sizes."""
import numpy as np

ERROR_KEYS = ("coord_epe_pct", "ad_pct", "local_distortion_pct", "staff_straight", "jp_baseline")
DEFAULT_POLICY = {"max_error_ratio": 1.05, "absolute_error_slack": 1e-6,
                  "max_coverage_drop": .005, "max_orientation_increase": .0005}


def geometry_gate(candidate, reference, policy=None):
    """Check each domain/kind separately; missing required metrics fail closed.

    Structure labels unavailable in a domain are omitted in its reference;
    they are never replaced by zero. Aggregate success cannot hide a failure.
    """
    policy = {**DEFAULT_POLICY, **(policy or {})}
    reasons = [] if reference else ["missing geometry reference"]
    for domain, expected in reference.items():
        actual = candidate.get(domain, {})
        for key in ERROR_KEYS:
            if expected.get(key) is None:
                continue
            if not np.isfinite(expected[key]) or expected[key] < 0:
                reasons.append(f"{domain}/{key}: invalid reference")
                continue
            value = actual.get(key)
            limit = expected[key] * policy["max_error_ratio"] + policy["absolute_error_slack"]
            if value is None or not np.isfinite(value) or value > limit:
                reasons.append(f"{domain}/{key}: {value} > {limit}")
        for key, sign, slack in (("coverage", -1, policy["max_coverage_drop"]),
                                 ("orientation_error_rate", 1, policy["max_orientation_increase"])):
            if expected.get(key) is None:
                continue
            if not np.isfinite(expected[key]):
                reasons.append(f"{domain}/{key}: invalid reference")
                continue
            value = actual.get(key)
            if value is None or not np.isfinite(value) or sign * (value - expected[key]) > slack:
                reasons.append(f"{domain}/{key}: missing or outside allowed change")
    return {"passed": not reasons, "reasons": reasons, "policy": policy}


def auxiliary_score(domains, objective):
    """Equal domain weight for classification; only annotated domains for pixels."""
    if objective == "kind":
        values = [d["kind"]["balanced_accuracy"] for d in domains.values()]
    elif objective == "pixels":
        values = [.5 * d["seg"]["miou"] + .5 * d["struct"]["macro_f1"]
                  for d in domains.values() if d.get("seg") and d.get("struct")]
    else:
        raise ValueError(objective)
    if not values or any(v is None or not np.isfinite(v) for v in values):
        raise ValueError("Missing validation supervision for selection")
    return float(np.mean(values))
