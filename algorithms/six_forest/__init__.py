"""
6Forest: IPv6 Target Generation via Space-Partitioning Forest.

Paper: "6Forest: An Ensemble Learning-based Approach to Target Generation
        for Internet-wide IPv6 Scanning"
Reference: https://github.com/tumi8/tma-23-target-generation/tree/main/algorithms/6Forest

Algorithm
---------
1. **Translate** — seeds are normalised to b4 (32 lowercase hex chars) and
   converted to numpy uint8 arrays where each element is one nibble (0-15).

2. **DHC partition** — a Dimensional Hierarchical Clustering tree is built
   with the *maxcovering* split strategy (prefer the dimension that contains
   the most repeated values) using a depth-first / LIFO queue.  Leaves are
   regions with fewer than ``min_region_size`` addresses (default 16).

3. **Outlier detection** — for each leaf region:
     a. *IsolatedForest* weighting: addresses that are alone in a nibble
        dimension accumulate higher isolation weight.
     b. *Four-Deviations* rule: recursively remove the heaviest address if
        its weight exceeds μ + 3σ of the remaining weights.
     c. The clean sub-set is refined further by ``iter_divide`` (DHC until
        splits would produce singletons) to yield tightly defined patterns.

4. **Iteration** (optional) — outlier addresses from step 3 are merged and
   re-processed through steps 2-3, up to ``max_iter`` additional passes.

5. **Generation** — patterns (wildcard strings) are sorted by descending seed
   density and expanded:
     * ≤3 wildcards  →  exhaustive cross-product
     * >3 wildcards  →  per-dimension cycling around each seed address

Usage
-----
    from algorithms.six_forest import run
    run(seeds="seeds.txt", output="targets.txt", budget=1_000_000)
"""

from __future__ import annotations

import numpy as np

from algorithms.six_tree.translation import normalize_to_b4
from .generation import generate_from_patterns
from .outliers import outlier_detect
from .partition import dhc_forest


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(seeds: str, output: str, budget: int) -> None:
    """Generate IPv6 target addresses using the 6Forest algorithm.

    Args:
        seeds:  Path to seed IPv6 address file (one per line).
        output: Path to write generated target addresses.
        budget: Maximum number of addresses to generate.
    """
    # ── Step 1: load and normalise seeds ─────────────────────────────────────
    print(f"[6Forest] Loading seeds from: {seeds}")
    b4_seeds = _load_seeds(seeds)
    if not b4_seeds:
        print("[6Forest] ERROR: no valid seed addresses — aborting.")
        return
    print(f"[6Forest] {len(b4_seeds):,} unique seeds loaded.")

    # Convert b4 strings → numpy nibble matrix (n × 32), dtype uint8.
    arrs = np.array(
        [[int(c, 16) for c in addr] for addr in b4_seeds], dtype=np.uint8
    )

    # ── Step 2: DHC partition ─────────────────────────────────────────────────
    print("[6Forest] Building DHC space partition …")
    regions = dhc_forest(arrs)
    print(f"[6Forest] {len(regions):,} leaf regions produced.")

    # ── Steps 3-4: outlier detection, optional re-iteration ──────────────────
    print("[6Forest] Running outlier detection …")
    patterns, residual_outliers = _collect_patterns(regions, max_iter=2)
    n_residual = sum(len(o) for o in residual_outliers)
    print(
        f"[6Forest] {len(patterns):,} patterns found, "
        f"{n_residual:,} residual outlier addresses."
    )

    # ── Step 5: generate targets ──────────────────────────────────────────────
    print(f"[6Forest] Generating up to {budget:,} target addresses …")
    targets = generate_from_patterns(patterns, budget)
    print(f"[6Forest] {len(targets):,} addresses generated.")

    # ── Write output ──────────────────────────────────────────────────────────
    with open(output, "w") as fh:
        for addr in targets:
            fh.write(addr + "\n")
    print(f"[6Forest] Targets written to: {output}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_patterns(
    regions: list[np.ndarray],
    max_iter: int = 2,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run outlier detection over regions, optionally re-iterating on outliers.

    Each iteration:
      1. Runs ``outlier_detect`` on every region in the current batch.
      2. Accumulates clean patterns.
      3. Merges all outlier arrays and re-runs DHC on them for the next pass.

    Args:
        regions:  Leaf regions from ``dhc_forest``.
        max_iter: Extra iterations over outlier sets (0 = single pass,
                  matching the original 6Forest paper).

    Returns:
        (all_patterns, final_outliers)
    """
    all_patterns: list[np.ndarray] = []
    pending = regions

    for iteration in range(1 + max_iter):
        next_outlier_arrays: list[np.ndarray] = []

        for region in pending:
            p, o = outlier_detect(region)
            all_patterns.extend(p)
            next_outlier_arrays.extend(o)

        if not next_outlier_arrays or iteration == max_iter:
            return all_patterns, next_outlier_arrays

        # Merge outliers and re-partition for the next iteration.
        valid = [o for o in next_outlier_arrays if len(o) > 0]
        if not valid:
            return all_patterns, []

        merged = np.vstack(valid)
        if len(merged) < 2:
            return all_patterns, [merged]

        pending = dhc_forest(merged)

    return all_patterns, []


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
