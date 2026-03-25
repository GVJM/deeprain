"""ar/scoring.py — AR-specific composite and combined scoring functions."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scoring import QUALITY_METRICS, compute_composite, _metric_array
from ar.rollout import COMBINED_TIER2_METRICS


def compute_combined_score(
    all_metrics: dict,
    tier2_metrics: dict,
) -> tuple:
    """
    Combined Tier1+Tier2 composite score for ranking AR models.

    Variants without Tier 2 data (no model.pt) get NaN for Tier 2 metrics and
    are ranked below rollout models by select_top_n_per_family.
    Delegates to compute_composite() — no scoring logic duplicated.
    """
    merged = {v: {**m, **tier2_metrics.get(v, {})} for v, m in all_metrics.items()}
    metric_defs = list(QUALITY_METRICS) + list(COMBINED_TIER2_METRICS)
    return compute_composite(merged, metric_defs=metric_defs)


def select_top_n_per_family(
    scores: dict,
    families: dict,
    n: int,
) -> list:
    """
    Return top N variants per family ranked by score (lower = better).
    NaN scores are ranked last within their family so they are excluded first.
    """
    from collections import defaultdict
    by_family: dict = defaultdict(list)
    for v, s in scores.items():
        by_family[families.get(v, v)].append(
            (s if not np.isnan(s) else float("inf"), v)
        )
    selected = []
    for fam_variants in by_family.values():
        fam_variants.sort(key=lambda x: x[0])
        selected.extend(v for _, v in fam_variants[:n])
    return selected
