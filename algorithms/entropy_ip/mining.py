"""Intra-segment pattern mining for entropy-ip.

Faithfully reproduces the a2-mining.py step from Foremski's entropy-ip
(IMC 2016, Akamai Technologies), using three detection passes:

  1. Heavy-hitter (const) values via IQR outlier detection.
  2. Dense clusters via DBSCAN — only when seg_bits >= 8.
  3. Contiguous value groups via gap-based segmentation (fallback).

Dependencies
------------
  numpy, sklearn.cluster.DBSCAN (already in requirements.txt)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Segment integer conversion
# ---------------------------------------------------------------------------

def seg_to_ints(arrs: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Convert nibble segment [start, stop) of every row to integer values.

    Args:
        arrs:  Nibble matrix (n, 32) uint8.
        start: First nibble index (inclusive).
        stop:  Last nibble index (exclusive).

    Returns:
        1-D int64 array of length n — one integer per seed address.
    """
    w = stop - start
    seg = arrs[:, start:stop].astype(np.int64)
    powers = np.array([16 ** (w - 1 - i) for i in range(w)], dtype=np.int64)
    return seg @ powers


# ---------------------------------------------------------------------------
# Pattern mining
# ---------------------------------------------------------------------------

def mine_segment(vals: np.ndarray, seg_bits: int) -> list[dict]:
    """Mine address patterns from one segment's integer values.

    Args:
        vals:     1-D array of integer segment values (one per seed address).
        seg_bits: Number of bits in the segment (= nibble_width * 4).

    Returns:
        List of pattern dicts:
          {"type": "const", "value": int,  "weight": float}
          {"type": "range", "lo": int, "hi": int, "weight": float}
        Weights sum approximately to 1.0 (a fallback range makes up the rest).
    """
    n = len(vals)
    if n == 0:
        return [{"type": "range", "lo": 0, "hi": max(0, 2 ** seg_bits - 1), "weight": 1.0}]

    patterns: list[dict] = []
    accounted: float = 0.0

    # ── Step 1: Heavy-hitter (const) values ──────────────────────────────────
    unique, counts = np.unique(vals, return_counts=True)

    if len(counts) >= 4:
        q1 = float(np.percentile(counts, 25))
        q3 = float(np.percentile(counts, 75))
        iqr = q3 - q1
        p_uniform = 1.0 / (2 ** seg_bits) if seg_bits < 50 else 0.0
        threshold = min(0.1 * n, max(q3 + 1.5 * iqr, p_uniform * n, 2.0))
    elif len(counts) == 1:
        # Single unique value → always a const
        threshold = 0.0
    else:
        # Very few unique values → use the maximum count
        threshold = float(counts.max())

    hh_values: set[int] = set()
    for v, c in zip(unique, counts):
        if c >= threshold:
            weight = c / n
            patterns.append({"type": "const", "value": int(v), "weight": weight})
            accounted += weight
            hh_values.add(int(v))

    remaining = vals[~np.isin(vals, list(hh_values))] if hh_values else vals.copy()

    if len(remaining) == 0:
        return _add_fallback(patterns, accounted, seg_bits)

    # ── Step 2: DBSCAN dense clusters (seg_bits >= 8 only) ───────────────────
    if seg_bits >= 8 and len(remaining) >= 5:
        from sklearn.cluster import DBSCAN  # noqa: PLC0415

        eps = (seg_bits / 4) ** 3
        X = remaining.reshape(-1, 1).astype(float)
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)

        clustered = np.zeros(len(remaining), dtype=bool)
        for label in set(labels):
            if label == -1:
                continue
            cmask = labels == label
            cvals = remaining[cmask]
            lo, hi = int(cvals.min()), int(cvals.max())
            span = hi - lo + 1
            # Density gate: observed count / expected-if-uniform >= 100
            expected = max(1e-12, n * span / (2 ** seg_bits))
            density = len(cvals) / expected
            if density >= 100:
                weight = len(cvals) / n
                patterns.append({"type": "range", "lo": lo, "hi": hi, "weight": weight})
                accounted += weight
                clustered |= cmask

        remaining = remaining[~clustered]

    if len(remaining) == 0:
        return _add_fallback(patterns, accounted, seg_bits)

    # ── Step 3: Contiguous value groups (gap-based) ───────────────────────────
    unique_rem = np.unique(remaining)

    if len(unique_rem) == 1:
        weight = len(remaining) / n
        patterns.append({"type": "const", "value": int(unique_rem[0]), "weight": weight})
        accounted += weight
    elif len(unique_rem) > 1:
        gaps = np.diff(unique_rem.astype(np.int64))
        # Split where gap > 4× the median inter-value gap
        median_gap = max(1.0, float(np.median(gaps)))
        split_pts = np.where(gaps > median_gap * 4)[0] + 1
        groups = np.split(unique_rem, split_pts)

        for g in groups:
            if len(g) == 0:
                continue
            lo, hi = int(g[0]), int(g[-1])
            cnt = int(np.sum((remaining >= lo) & (remaining <= hi)))
            if cnt == 0:
                continue
            weight = cnt / n
            if lo == hi:
                patterns.append({"type": "const", "value": lo, "weight": weight})
            else:
                patterns.append({"type": "range", "lo": lo, "hi": hi, "weight": weight})
            accounted += weight

    return _add_fallback(patterns, accounted, seg_bits)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _add_fallback(
    patterns: list[dict],
    accounted: float,
    seg_bits: int,
) -> list[dict]:
    """Append a wide fallback range pattern for any unaccounted probability mass."""
    leftover = 1.0 - accounted
    if leftover > 0.01 and seg_bits > 0:
        max_val = 2 ** seg_bits - 1
        patterns.append({"type": "range", "lo": 0, "hi": max_val, "weight": leftover})
    return patterns
