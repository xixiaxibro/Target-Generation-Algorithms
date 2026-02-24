"""Entropy-based segmentation of IPv6 address nibble positions.

Faithfully reproduces the a1-segments.py step from Foremski's entropy-ip
(IMC 2016, Akamai Technologies).

Segmentation rules
------------------
- nibbles 0..(isp_boundary-1)  : one segment — ISP prefix, never sub-divided
- nibbles isp_boundary..(net_boundary-1) : one segment — network ID
- nibbles net_boundary..31 : split where adjacent entropy tier differs AND
  |H[d] - H[d-1]| > 0.05

Entropy tiers (log2, range [0, 4] for 4-bit nibbles)
------------------------------------------------------
  t1: H < 0.025   t2: H < 0.1   t3: H < 0.3
  t4: H < 0.5     t5: H < 0.9   t6: H >= 0.9
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _entropy(col: np.ndarray) -> float:
    """Shannon entropy (log2) for one nibble column."""
    counts = np.bincount(col.astype(np.intp), minlength=16)
    total = len(col)
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def _classify(h: float) -> int:
    """Map entropy value to tier 1–6."""
    if h < 0.025:
        return 1
    if h < 0.1:
        return 2
    if h < 0.3:
        return 3
    if h < 0.5:
        return 4
    if h < 0.9:
        return 5
    return 6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_segments(
    arrs: np.ndarray,
    isp_boundary: int = 8,
    net_boundary: int = 16,
) -> list[dict]:
    """Find entropy-based segment boundaries across the 32 nibble positions.

    Args:
        arrs:          Seed nibble matrix (n, 32) uint8.
        isp_boundary:  Nibble index where ISP prefix ends (exclusive).
                       Nibbles 0..(isp_boundary-1) form one forced segment.
        net_boundary:  Nibble index where network ID ends (exclusive).
                       Nibbles isp_boundary..(net_boundary-1) form one forced
                       segment; tier-change splits apply only beyond this.

    Returns:
        List of {"start": int, "stop": int} half-open interval dicts,
        one per segment, covering [0, 32) without gaps or overlaps.
    """
    entropies = [_entropy(arrs[:, d]) for d in range(32)]
    tiers = [_classify(h) for h in entropies]

    # Fixed mandatory boundaries
    boundaries: set[int] = {0, isp_boundary, net_boundary, 32}

    # Tier-change splits only in the interface portion (beyond net_boundary)
    for d in range(net_boundary + 1, 32):
        if (
            tiers[d] != tiers[d - 1]
            and abs(entropies[d] - entropies[d - 1]) > 0.05
        ):
            boundaries.add(d)

    sorted_bounds = sorted(boundaries)
    return [
        {"start": sorted_bounds[i], "stop": sorted_bounds[i + 1]}
        for i in range(len(sorted_bounds) - 1)
    ]
