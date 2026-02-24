"""
DHC space partitioning for 6Forest.

Differences from 6tree's DHC
-----------------------------
* Split strategy: **maxcovering** (picks the dimension that maximises the sum
  of non-singleton value counts) rather than leftmost.  A tiebreak rule keeps
  the leftmost dimension when the gain does not justify the rightward shift.
* Traversal: **depth-first / LIFO** queue (LifoQueue) rather than recursion.
* Leaf threshold: regions with fewer than `min_region_size` addresses (default
  16) are returned as leaves rather than split further.

These choices match the original 6Forest implementation exactly.
"""

from __future__ import annotations

import queue
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Internal split function
# ---------------------------------------------------------------------------

def _maxcovering_split(arrs: np.ndarray) -> List[np.ndarray]:
    """Choose the best split dimension and return one sub-array per group.

    The *maxcovering* score for dimension *i* is the sum of the counts of
    nibble values that appear more than once:
        score(i) = Σ  count(v)  for v where count(v) > 1

    Leftmost dimension is preferred unless a right-side dimension scores
    strictly more than (gap_in_position) better.

    Returns a list of sub-arrays.  Returns ``[arrs]`` (no actual split)
    when every dimension is uniform — callers must guard against this to
    avoid infinite loops.
    """
    Tarrs = arrs.T          # shape (32, n)
    covering: List[int] = []

    leftmost_idx = -1
    leftmost_cov = -1

    for i in range(32):
        counts = np.bincount(Tarrs[i], minlength=16)
        if np.count_nonzero(counts) == 1:
            covering.append(-1)          # fixed dimension, skip
        else:
            c = int(np.sum(counts[counts > 1]))
            covering.append(c)
            if leftmost_idx == -1:
                leftmost_idx = i
                leftmost_cov = c

    if leftmost_idx == -1:
        # All dimensions fixed — every address is identical; can't split.
        return [arrs]

    best_idx = int(np.argmax(covering))
    best_cov = covering[best_idx]

    # Prefer leftmost unless rightward gain exceeds positional cost.
    if best_cov - leftmost_cov <= best_idx - leftmost_idx:
        best_idx = leftmost_idx

    counts     = np.bincount(Tarrs[best_idx], minlength=16)
    nibble_vals = np.argwhere(counts).reshape(-1)

    return [arrs[Tarrs[best_idx] == nib] for nib in nibble_vals]


# ---------------------------------------------------------------------------
# Public DHC builder
# ---------------------------------------------------------------------------

def dhc_forest(
    arrs: np.ndarray,
    min_region_size: int = 16,
) -> List[np.ndarray]:
    """Build a space-partitioning forest via DHC with maxcovering splits.

    Uses a depth-first (LIFO) queue to partition the seed address set
    recursively.  A node becomes a leaf when:
      * it contains fewer than ``min_region_size`` addresses, or
      * no split can reduce the region (all addresses identical).

    Args:
        arrs:             Numpy array of shape (n, 32), dtype uint8.
                          Each row is one seed address as 32 hex nibbles.
        min_region_size:  Minimum addresses required to attempt a split.

    Returns:
        List of leaf region arrays, each of shape (k, 32).
    """
    q: queue.LifoQueue = queue.LifoQueue()
    q.put(arrs)
    regions: List[np.ndarray] = []

    while not q.empty():
        region = q.get()

        if len(region) < min_region_size:
            regions.append(region)
            continue

        splits = _maxcovering_split(region)

        if len(splits) == 1:
            # No actual split possible — treat as leaf.
            regions.append(region)
            continue

        for s in splits:
            q.put(s)

    return regions
