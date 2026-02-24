"""
DHC space partitioning for 6Graph.

Differences from 6Forest's DHC
--------------------------------
* Split strategy: **leftmost** — scan dimensions left-to-right and split on
  the first dimension that has more than one distinct nibble value.
* Traversal: **breadth-first / FIFO** queue (Queue) rather than LIFO.
* Leaf threshold: regions with fewer than ``min_region_size`` addresses
  (default 16) are returned as leaves.

These choices match the original 6Graph / SpacePartition.py implementation.
"""

from __future__ import annotations

import queue
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Internal split function
# ---------------------------------------------------------------------------

def _leftmost_split(arrs: np.ndarray) -> List[np.ndarray]:
    """Find the leftmost dimension with >1 distinct nibble value and split.

    Returns a list of sub-arrays, one per distinct nibble value in that
    dimension.  Returns ``[arrs]`` (no actual split) when every dimension is
    uniform — callers must guard against this to avoid infinite loops.
    """
    for dim in range(32):
        unique = np.unique(arrs[:, dim])
        if len(unique) > 1:
            return [arrs[arrs[:, dim] == v] for v in unique]
    # All dimensions uniform — every address is identical; can't split.
    return [arrs]


# ---------------------------------------------------------------------------
# Public DHC builder
# ---------------------------------------------------------------------------

def dhc_graph(
    arrs: np.ndarray,
    min_region_size: int = 16,
) -> List[np.ndarray]:
    """Build a space partition via DHC with leftmost splits and BFS traversal.

    Uses a breadth-first (FIFO) queue to partition the seed address set.
    A node becomes a leaf when:
      * it contains fewer than ``min_region_size`` addresses, or
      * no split can reduce the region (all addresses identical).

    Args:
        arrs:             Numpy array of shape (n, 32), dtype uint8.
                          Each row is one seed address as 32 hex nibbles.
        min_region_size:  Minimum addresses required to attempt a split.

    Returns:
        List of leaf region arrays, each of shape (k, 32).
    """
    q: queue.Queue = queue.Queue()   # FIFO — BFS traversal
    q.put(arrs)
    regions: List[np.ndarray] = []

    while not q.empty():
        region = q.get()

        if len(region) <= min_region_size:
            regions.append(region)
            continue

        splits = _leftmost_split(region)

        if len(splits) == 1:
            # No actual split possible — treat as leaf.
            regions.append(region)
            continue

        for s in splits:
            q.put(s)

    return regions
