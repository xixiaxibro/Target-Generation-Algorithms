"""
6Graph: Graph Pattern Mining TGA.

Paper: "Target Acquired? Evaluating Target Generation Algorithms for IPv6"
       (TMA 2023)
Reference: https://github.com/tumi8/tma-23-target-generation/tree/main/algorithms

Algorithm
---------
1. **Translate** — seeds are normalised to b4 (32 lowercase hex chars) and
   converted to numpy uint8 arrays where each element is one nibble (0-15).

2. **DHC partition** — a Dimensional Hierarchical Clustering tree is built
   with the *leftmost* split strategy (split on the first dimension that has
   >1 distinct nibble value) using a breadth-first / FIFO queue.  Leaves are
   regions with no more than ``min_region_size`` addresses (default 16).

3. **Graph clustering** — for each leaf region:
     a. Build a graph with one node per address.
     b. Add edges between pairs with nibble Hamming distance ≤ 12.
     c. Apply a density gate: reject any edge that would push either
        endpoint's neighbourhood density below 0.8.
     d. Extract connected components: size > 1 → cluster, size == 1 → outlier.

4. **Iteration** — outlier addresses are merged and re-processed through
   steps 2-3, up to ``max_iter`` additional passes (default 3).  Remaining
   outliers after all iterations are kept as singleton patterns.

5. **Generation** — patterns (wildcard strings) are sorted by descending seed
   density and expanded:
     * ≤3 wildcards  →  exhaustive cross-product
     * >3 wildcards  →  per-dimension cycling around each seed address

Usage
-----
    from algorithms.six_graph import run
    run(seeds="seeds.txt", output="targets.txt", budget=1_000_000)
"""

from __future__ import annotations

import numpy as np

from algorithms.six_tree.translation import normalize_to_b4
from .generation import generate_from_patterns
from .graph import cluster_by_graph
from .partition import dhc_graph


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(seeds: str, output: str, budget: int) -> None:
    """Generate IPv6 target addresses using the 6Graph algorithm.

    Args:
        seeds:  Path to seed IPv6 address file (one per line).
        output: Path to write generated target addresses.
        budget: Maximum number of addresses to generate.
    """
    # ── Step 1: load and normalise seeds ─────────────────────────────────────
    print(f"[6Graph] Loading seeds from: {seeds}")
    b4_seeds = _load_seeds(seeds)
    if not b4_seeds:
        print("[6Graph] ERROR: no valid seed addresses — aborting.")
        return
    print(f"[6Graph] {len(b4_seeds):,} unique seeds loaded.")

    # Convert b4 strings → numpy nibble matrix (n × 32), dtype uint8.
    arrs = np.array(
        [[int(c, 16) for c in addr] for addr in b4_seeds], dtype=np.uint8
    )

    # ── Step 2: DHC partition ─────────────────────────────────────────────────
    print("[6Graph] Building DHC space partition (leftmost-split, BFS) …")
    regions = dhc_graph(arrs)
    print(f"[6Graph] {len(regions):,} leaf regions produced.")

    # ── Steps 3-4: graph clustering with optional re-iteration ───────────────
    print("[6Graph] Running graph clustering …")
    clusters = _collect_clusters(regions, max_iter=3)
    print(f"[6Graph] {len(clusters):,} clusters/patterns collected.")

    # ── Step 5: generate targets ──────────────────────────────────────────────
    print(f"[6Graph] Generating up to {budget:,} target addresses …")
    targets = generate_from_patterns(clusters, budget)
    print(f"[6Graph] {len(targets):,} addresses generated.")

    # ── Write output ──────────────────────────────────────────────────────────
    with open(output, "w") as fh:
        for addr in targets:
            fh.write(addr + "\n")
    print(f"[6Graph] Targets written to: {output}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_clusters(
    regions: list[np.ndarray],
    max_iter: int = 3,
) -> list[np.ndarray]:
    """Run graph clustering over regions, re-iterating on outliers.

    Each pass:
      1. Runs ``cluster_by_graph`` on every region in the current batch.
      2. Accumulates clusters.
      3. Merges all outlier arrays and re-runs DHC on them for the next pass.

    Remaining outliers after all iterations are added as singleton patterns
    so that no seed is silently discarded.

    Args:
        regions:  Leaf regions from ``dhc_graph``.
        max_iter: Number of re-iteration passes over outlier sets.

    Returns:
        List of cluster arrays (each a (k, 32) uint8 array).
    """
    clusters: list[np.ndarray] = []
    pending: list[np.ndarray] = []

    # First pass over initial regions.
    for region in regions:
        c, o = cluster_by_graph(region)
        clusters.extend(c)
        pending.extend(o)

    # Re-iteration passes.
    for _ in range(max_iter):
        if not pending:
            break

        valid = [o for o in pending if len(o) > 0]
        if not valid:
            break

        merged = np.vstack(valid)
        if len(merged) < 2:
            # Can't partition further; keep as-is.
            clusters.extend(valid)
            pending = []
            break

        new_regions = dhc_graph(merged)
        pending = []

        for region in new_regions:
            c, o = cluster_by_graph(region)
            clusters.extend(c)
            pending.extend(o)

    # Any remaining outliers become singleton patterns.
    clusters.extend(o for o in pending if len(o) > 0)

    return clusters


def _load_seeds(path: str) -> list[str]:
    """Read seed file and return sorted unique b4-format addresses."""
    b4_set: set[str] = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            b4 = normalize_to_b4(line)
            if b4 is not None:
                b4_set.add(b4)
    return sorted(b4_set)
