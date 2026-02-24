"""
DET: Density Estimation Tree — IPv6 Target Generation Algorithm.

Paper: "DET: Enabling Efficient Probing of IPv6 Active Addresses"
Reference: https://github.com/tumi8/tma-23-target-generation/tree/main/algorithms/DET

Algorithm
---------
1. **Translate** — seeds are normalised to b4 (32 lowercase hex chars) and
   converted to numpy uint8 arrays where each element is one nibble (0-15).

2. **DHC partition** — a Dimensional Hierarchical Clustering tree is built
   with the *minimum-entropy* split strategy: at each node the dimension
   whose Shannon entropy is smallest (but non-zero) is chosen for splitting.
   Low entropy in a dimension means most addresses share the same nibble value
   there, so splitting produces tight, dense sub-groups.  Traversal is
   depth-first / LIFO, matching the original recursive DET implementation.
   Leaves are regions with ≤ ``min_region_size`` addresses (default 16).

3. **Generation** — leaf regions become wildcard patterns; dimensions with a
   single unique nibble value are fixed, others become '*'.  Patterns are
   sorted by descending seed density and expanded:
     * ≤3 wildcards  →  exhaustive cross-product
     * >3 wildcards  →  per-dimension cycling around each seed address

   There is no outlier-detection or graph-clustering step — DET's online
   scanning feedback loop is replaced here by offline density-ranked expansion.

Usage
-----
    from algorithms.det import run
    run(seeds="seeds.txt", output="targets.txt", budget=1_000_000)
"""

from __future__ import annotations

import numpy as np

from algorithms.six_tree.translation import normalize_to_b4
from .generation import generate_from_patterns
from .partition import dhc_det


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(seeds: str, output: str, budget: int) -> None:
    """Generate IPv6 target addresses using the DET algorithm.

    Args:
        seeds:  Path to seed IPv6 address file (one per line).
        output: Path to write generated target addresses.
        budget: Maximum number of addresses to generate.
    """
    # ── Step 1: load and normalise seeds ─────────────────────────────────────
    print(f"[DET] Loading seeds from: {seeds}")
    b4_seeds = _load_seeds(seeds)
    if not b4_seeds:
        print("[DET] ERROR: no valid seed addresses — aborting.")
        return
    print(f"[DET] {len(b4_seeds):,} unique seeds loaded.")

    # Convert b4 strings → numpy nibble matrix (n × 32), dtype uint8.
    arrs = np.array(
        [[int(c, 16) for c in addr] for addr in b4_seeds], dtype=np.uint8
    )

    # ── Step 2: DHC partition (minimum-entropy split, DFS) ───────────────────
    print("[DET] Building entropy space tree …")
    regions = dhc_det(arrs)
    print(f"[DET] {len(regions):,} leaf regions produced.")

    # ── Step 3: generate targets from leaf patterns ──────────────────────────
    print(f"[DET] Generating up to {budget:,} target addresses …")
    targets = generate_from_patterns(regions, budget)
    print(f"[DET] {len(targets):,} addresses generated.")

    # ── Write output ──────────────────────────────────────────────────────────
    with open(output, "w") as fh:
        for addr in targets:
            fh.write(addr + "\n")
    print(f"[DET] Targets written to: {output}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
