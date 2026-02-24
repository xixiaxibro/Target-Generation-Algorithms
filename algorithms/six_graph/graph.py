"""
Graph-based clustering for 6Graph.

Faithfully reproduces the PatternMining.py logic from the original
tma-23-target-generation repository:

1. Build a graph over the regions returned by DHC, with one node per region.
2. Enumerate all pairs (i, j) of regions; compute nibble Hamming distance
   between their representative nibble arrays.
3. Sort candidate edges by distance (ascending, Kruskal-style).
4. Greedily add each edge if both endpoints' neighbourhood density remains
   above ``min_density`` (density gate).
5. Extract connected components: size > 1 → cluster, size == 1 → outlier.

The density metric used here matches PatternMining.py::
    density = Σ_d len(unique(arrs[:, d])) / (n * 32)
A low value means addresses are tightly packed (few distinct nibbles per
dimension), which is what we want in a cluster.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "networkx is required for 6Graph clustering. "
        "Install it with: pip install networkx"
    ) from exc


# ---------------------------------------------------------------------------
# Density helper
# ---------------------------------------------------------------------------

def _density(arrs: np.ndarray) -> float:
    """Compute intra-cluster density.

    Lower = addresses are more similar to each other (good cluster quality).
    A perfectly uniform set (all addresses identical) has density 1/n.

    Formula::
        density = Σ_d |unique(arrs[:, d])| / (n * 32)
    """
    n = len(arrs)
    if n == 0:
        return 0.0
    distinct_count = sum(len(np.unique(arrs[:, d])) for d in range(32))
    return distinct_count / (n * 32)


# ---------------------------------------------------------------------------
# Pairwise nibble Hamming distance
# ---------------------------------------------------------------------------

def _nibble_hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Count the number of nibble positions where two addresses differ."""
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------------------
# Public clustering function
# ---------------------------------------------------------------------------

def cluster_by_graph(
    arrs: np.ndarray,
    max_distance: int = 12,
    min_density: float = 0.8,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Cluster a set of nibble-encoded addresses via greedy graph edge-adding.

    Algorithm
    ---------
    1. Treat each address as a node.
    2. Enumerate all pairs; compute nibble Hamming distance.
    3. Keep only pairs with distance ≤ ``max_distance``.
    4. Sort by distance ascending (Kruskal-style).
    5. For each candidate edge (i, j):
       a. Tentatively add the edge to the graph.
       b. Collect the nibble arrays of *all neighbours* of i (resp. j).
       c. Compute density of those neighbour sets.
       d. If either density < ``min_density`` → remove the edge (density gate).
    6. Extract connected components:
       * size > 1  → cluster  (np.ndarray of addresses in that component)
       * size == 1 → outlier  (np.ndarray of a single address)

    Args:
        arrs:         (n, 32) uint8 nibble matrix.
        max_distance: Maximum nibble Hamming distance to consider an edge.
        min_density:  Minimum density threshold; edges that would push either
                      endpoint's neighbourhood below this are rejected.

    Returns:
        (clusters, outliers) — two lists of numpy arrays.
    """
    n = len(arrs)

    # Edge case: single address.
    if n == 1:
        return [], [arrs]

    # ── Build candidate edges ──────────────────────────────────────────────
    edges: list[tuple[int, int, int]] = []   # (distance, i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = _nibble_hamming(arrs[i], arrs[j])
            if dist <= max_distance:
                edges.append((dist, i, j))

    # No edges satisfy the distance constraint → all are outliers.
    if not edges:
        return [], [arrs[idx:idx+1] for idx in range(n)]

    edges.sort()   # ascending by distance

    # ── Greedy edge-adding with density gate ───────────────────────────────
    G: nx.Graph = nx.Graph()
    G.add_nodes_from(range(n))

    for _dist, i, j in edges:
        G.add_edge(i, j)

        # Check density of neighbourhood of i (include i itself).
        nbrs_i = list(G.neighbors(i)) + [i]
        nbrs_j = list(G.neighbors(j)) + [j]

        dens_i = _density(arrs[nbrs_i])
        dens_j = _density(arrs[nbrs_j])

        if dens_i < min_density or dens_j < min_density:
            G.remove_edge(i, j)   # density gate: reject this edge

    # ── Extract clusters and outliers ──────────────────────────────────────
    clusters: List[np.ndarray] = []
    outliers: List[np.ndarray] = []

    for component in nx.connected_components(G):
        indices = sorted(component)
        sub = arrs[indices]
        if len(component) > 1:
            clusters.append(sub)
        else:
            outliers.append(sub)

    return clusters, outliers
