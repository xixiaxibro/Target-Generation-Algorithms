"""
DHC space partitioning for DET.

Differences from 6Forest and 6Graph
-------------------------------------
* Split strategy: **minimum-entropy** — scan all 32 nibble dimensions and
  pick the one whose Shannon entropy is smallest (but non-zero).  A dimension
  with low entropy has a highly skewed value distribution, meaning most
  addresses share the same nibble there — splitting on it creates tight, dense
  sub-groups.  Dimensions with entropy == 0 (all addresses identical in that
  dimension) are skipped.
* Traversal: **depth-first / LIFO**, matching the recursive structure of the
  original DHC.py from the DET reference implementation.
* Leaf threshold: regions with ≤ ``min_region_size`` addresses (default 16)
  are returned as leaves without further splitting.

This matches the original ``DHC()`` / ``get_splitP()`` logic from
https://github.com/tumi8/tma-23-target-generation/tree/main/algorithms/DET
"""

from __future__ import annotations

import queue
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Internal split function
# ---------------------------------------------------------------------------

def _min_entropy_split(arrs: np.ndarray) -> List[np.ndarray]:
    """Choose the split dimension with minimum non-zero Shannon entropy.

    Shannon entropy for dimension *d*::
        H(d) = -Σ  p_i * ln(p_i)   (natural log, matching the original)

    Dimensions where all addresses share the same nibble value (H == 0) are
    skipped.  If every dimension is uniform the region cannot be split.

    Returns a list of sub-arrays, one per distinct nibble value in the chosen
    dimension.  Returns ``[arrs]`` when no split is possible.
    """
    n = len(arrs)
    best_entropy = float("inf")
    best_dim = -1

    for d in range(32):
        counts = np.bincount(arrs[:, d], minlength=16)
        nonzero = counts[counts > 0]
        if len(nonzero) == 1:
            continue                         # entropy == 0, uniform → skip
        probs = nonzero / n
        entropy = -float(np.sum(probs * np.log(probs)))   # natural log
        if entropy < best_entropy:
            best_entropy = entropy
            best_dim = d

    if best_dim == -1:
        # All dimensions uniform — every address is identical; can't split.
        return [arrs]

    values = np.unique(arrs[:, best_dim])
    return [arrs[arrs[:, best_dim] == v] for v in values]


# ---------------------------------------------------------------------------
# Public DHC builder
# ---------------------------------------------------------------------------

def dhc_det(
    arrs: np.ndarray,
    min_region_size: int = 16,
) -> List[np.ndarray]:
    """Build a space partition via DHC with minimum-entropy splits.

    Uses a depth-first (LIFO) queue, matching the recursive structure of the
    original DET implementation.  A node becomes a leaf when:
      * it contains ≤ ``min_region_size`` addresses, or
      * no split can reduce the region (all addresses identical in every dim).

    Args:
        arrs:             Numpy array of shape (n, 32), dtype uint8.
                          Each row is one seed address as 32 hex nibbles.
        min_region_size:  Maximum number of addresses allowed in a leaf.

    Returns:
        List of leaf region arrays, each of shape (k, 32).
    """
    q: queue.LifoQueue = queue.LifoQueue()   # DFS — matches original recursion
    q.put(arrs)
    regions: List[np.ndarray] = []

    while not q.empty():
        region = q.get()

        if len(region) <= min_region_size:
            regions.append(region)
            continue

        splits = _min_entropy_split(region)

        if len(splits) == 1:
            # No actual split possible — treat as leaf.
            regions.append(region)
            continue

        for s in splits:
            q.put(s)

    return regions
