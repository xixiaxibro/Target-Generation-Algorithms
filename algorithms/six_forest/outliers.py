"""
Outlier detection for 6Forest.

Two-stage process
-----------------
1. **IsolatedForest weighting**: for every free (non-fixed) dimension, any
   address that is the *only* address with that nibble value gets its
   weight incremented by 1 / (number of singletons in that dimension).
   Addresses that are isolated in many dimensions accumulate high weights.

2. **Four-Deviations (4D) rule**: the highest-weight address is declared an
   outlier if  max_weight - μ_rest  >  3 * σ_rest.  The procedure is applied
   recursively until no more outliers are found.

After outlier removal, the remaining clean addresses are further refined by
``_iter_divide``: DHC subdivision continues until any split would produce a
singleton group.  The result is the final list of patterns.

Outliers are returned separately so the caller can optionally feed them back
into another DHC + OutlierDetect iteration.
"""

from __future__ import annotations

import queue
from typing import List, Tuple

import numpy as np

from .partition import _maxcovering_split


# ---------------------------------------------------------------------------
# IsolatedForest weighting
# ---------------------------------------------------------------------------

def _isolated_forest_weights(arrs: np.ndarray) -> np.ndarray:
    """Return per-address isolation weight vector.

    Each address that appears alone in a free dimension (singleton) receives
    weight  1 / (number of singletons in that dimension).
    """
    Tarrs = arrs.T
    weights = np.zeros(len(arrs), dtype=float)

    for i in range(32):
        counts = np.bincount(Tarrs[i], minlength=16)
        singleton_mask = counts == 1
        n_singletons = int(np.sum(singleton_mask))
        if n_singletons == 0:
            continue
        for nib in np.argwhere(singleton_mask).reshape(-1):
            idx = int(np.argwhere(Tarrs[i] == nib).reshape(-1)[0])
            weights[idx] += 1.0 / n_singletons

    return weights


# ---------------------------------------------------------------------------
# Four-Deviations rule
# ---------------------------------------------------------------------------

def _four_d(weights: List[float]) -> List[float]:
    """Recursively find outlier weights via the 3-sigma rule.

    Returns a list of weight values identified as outliers.
    The list may be empty if no outlier is found.
    """
    if len(weights) <= 2:
        return []

    outlier_idx = int(np.argmax(weights))
    outlier_val = float(weights[outlier_idx])

    rest = list(weights)
    rest.pop(outlier_idx)

    avg_rest = float(np.average(rest))
    std_rest = float(np.sqrt(np.var(rest)))      # population std dev

    if outlier_val - avg_rest > 3 * std_rest:
        return [outlier_val] + _four_d(rest)
    else:
        return []


# ---------------------------------------------------------------------------
# iter_divide — post-outlier-removal refinement
# ---------------------------------------------------------------------------

def _iter_divide(arrs: np.ndarray) -> List[np.ndarray]:
    """Refine a clean region by continuing DHC until splits produce singletons.

    Unlike the main ``dhc_forest`` (which stops at a size threshold), this
    function stops when *any* of the resulting split groups would be a singleton
    (size == 1).  This ensures the returned patterns contain only cohesive
    clusters, not isolated addresses.
    """
    q: queue.LifoQueue = queue.LifoQueue()
    q.put(arrs)
    regions: List[np.ndarray] = []

    while not q.empty():
        region = q.get()
        splits = _maxcovering_split(region)

        # Stop if no real split exists or any split would create a singleton.
        if len(splits) == 1 or any(len(s) == 1 for s in splits):
            regions.append(region)
        else:
            for s in splits:
                q.put(s)

    return regions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def outlier_detect(
    arrs: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Detect and remove outlier addresses from a DHC leaf region.

    Args:
        arrs: Shape (n, 32) uint8 array — all addresses in one leaf region.

    Returns:
        patterns:  List of refined sub-region arrays (clean clusters).
                   Each pattern array can be converted to a wildcard string.
        outliers:  List of outlier address arrays for optional re-iteration.
                   Usually a single array; may be empty if no outliers found.
    """
    n = len(arrs)

    # ── Trivial cases ────────────────────────────────────────────────────────
    if n == 1:
        return [], [arrs]

    if n == 2:
        dist = int(np.sum(arrs[0] != arrs[1]))
        if dist > 12:
            # Two very dissimilar addresses — both are outliers.
            return [], [arrs]
        else:
            return [arrs], []

    # ── Weight addresses by isolation ────────────────────────────────────────
    weights = _isolated_forest_weights(arrs)

    # ── Find outlier indices ─────────────────────────────────────────────────
    outlier_weight_vals = _four_d(list(weights))

    if not outlier_weight_vals:
        # No outliers — refine and return all as patterns.
        return _iter_divide(arrs), []

    # Remove outliers one by one in weight order to avoid index conflicts.
    outlier_indices: set[int] = set()
    for ow in outlier_weight_vals:
        candidates = np.where(weights == ow)[0]
        for idx in candidates:
            if int(idx) not in outlier_indices:
                outlier_indices.add(int(idx))
                break

    clean_indices = [i for i in range(n) if i not in outlier_indices]
    if not clean_indices:
        return [], [arrs]

    clean    = arrs[clean_indices]
    outliers = arrs[sorted(outlier_indices)]

    # ── Refine clean region ───────────────────────────────────────────────────
    patterns = _iter_divide(clean)

    return patterns, [outliers]
