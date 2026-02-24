"""
6Scan: systematic IPv6 neighbourhood probing for offline target generation.

Paper: "Clusters in the Expanse: Understanding and Unbiasing IPv6 Hitlists"
       Gasser et al., IMC 2018.  The scanning strategy is used in TMA-23 as
       a target-generation baseline.
Reference: https://github.com/tumi8/tma-23-target-generation/tree/main/algorithms/6Scan

Algorithm (offline variant — no active-probing feedback loop)
-------------------------------------------------------------
The online 6Scan expands from responsive addresses by probing their IID
neighbours and repeating with every new hit.  Without a live scanner we
replicate the *generation* layer faithfully:

1. **Load** — seeds normalised to b4 nibble matrix.
2. **Group** — seeds are bucketed by /64 prefix (first 16 nibbles).
   Larger groups represent more-responsive subnets and are explored first.
3. **Enumerate** — for each /64 group, IID candidates are emitted in BFS
   order of Hamming distance from the nearest seed:
     • distance 1: vary exactly one of the 16 IID nibbles  (≤ 240/seed)
     • distance 2: vary exactly two IID nibbles            (≤ 27 000/seed)
     • distance 3: vary exactly three IID nibbles          (≤ 1.5 M/seed)
   Distances beyond 3 fall back to uniform IID sampling to fill the budget.
4. **Deduplicate** and stop once *budget* unique addresses are collected.

Usage
-----
    from algorithms.six_scan import run
    run(seeds="seeds.txt", output="targets.txt", budget=1_000_000)
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict

import numpy as np

from algorithms.six_tree.translation import b4_to_std, normalize_to_b4

_HEX       = "0123456789abcdef"
_MAX_HAMMING = 3    # maximum Hamming distance explored exhaustively
_IID_START   = 16   # nibble index where the IID begins (positions 16–31)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(seeds: str, output: str, budget: int) -> None:
    """Generate IPv6 target addresses using the 6Scan neighbourhood strategy.

    Args:
        seeds:  Path to seed IPv6 address file (one per line).
        output: Path to write generated target addresses.
        budget: Maximum number of addresses to generate.
    """
    print(f"[6Scan] Loading seeds from: {seeds}")
    b4_seeds = _load_seeds(seeds)
    if not b4_seeds:
        print("[6Scan] ERROR: no valid seed addresses — aborting.")
        return
    print(f"[6Scan] {len(b4_seeds):,} unique seeds loaded.")

    arrs = np.array([[int(c, 16) for c in addr] for addr in b4_seeds], dtype=np.uint8)

    # Bucket seeds by /64 prefix (first 16 nibbles)
    groups: dict[tuple, list[list[int]]] = defaultdict(list)
    for row in arrs:
        key = tuple(row[:_IID_START].tolist())
        groups[key].append(row.tolist())

    # Largest /64 groups first (more seeds → higher likelihood of neighbours)
    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    print(f"[6Scan] {len(sorted_groups):,} unique /64 prefixes found.")
    print(f"[6Scan] Enumerating neighbourhood up to Hamming distance {_MAX_HAMMING} …")

    targets = _enumerate_targets(sorted_groups, budget)
    print(f"[6Scan] {len(targets):,} addresses generated.")

    with open(output, "w") as fh:
        for addr in targets:
            fh.write(addr + "\n")
    print(f"[6Scan] Targets written to: {output}")


# ---------------------------------------------------------------------------
# Neighbourhood enumeration
# ---------------------------------------------------------------------------

def _enumerate_targets(
    sorted_groups: list[list[list[int]]],
    budget: int,
) -> list[str]:
    """Emit target addresses across all /64 groups in BFS distance order.

    Sweeps distance 1, 2, 3 across all groups before advancing to the next
    distance, so that candidates closest to any seed are emitted first.
    """
    results: list[str] = []
    seen: set[str] = set()

    # Passes 1–3: exhaustive Hamming-distance enumeration
    for dist in range(1, _MAX_HAMMING + 1):
        if len(results) >= budget:
            break
        for group_seeds in sorted_groups:
            if len(results) >= budget:
                break
            for seed in group_seeds:
                if len(results) >= budget:
                    return results
                for b4 in _iid_variations(seed, dist):
                    if b4 not in seen:
                        seen.add(b4)
                        results.append(b4_to_std(b4))
                        if len(results) >= budget:
                            return results

    # Pass 4: random IID fill for any remaining budget
    if len(results) < budget:
        for group_seeds in sorted_groups:
            if len(results) >= budget:
                break
            prefix = group_seeds[0][:_IID_START]
            needed = budget - len(results)
            for _ in range(needed * 2):   # oversample slightly to account for collisions
                iid = [random.randint(0, 15) for _ in range(16)]
                b4  = "".join(_HEX[n] for n in prefix + iid)
                if b4 not in seen:
                    seen.add(b4)
                    results.append(b4_to_std(b4))
                if len(results) >= budget:
                    break

    return results


def _iid_variations(seed: list[int], hamming_dist: int):
    """Yield b4 strings differing from *seed* in exactly *hamming_dist* IID nibbles.

    Only positions 16–31 (the IID) are varied; the /64 prefix is kept intact.
    """
    prefix = seed[:_IID_START]
    iid    = seed[_IID_START:]

    for positions in itertools.combinations(range(16), hamming_dist):
        # All alternatives at the chosen positions (exclude the original value)
        alt_ranges = [
            [v for v in range(16) if v != iid[p]]
            for p in positions
        ]
        for values in itertools.product(*alt_ranges):
            new_iid = list(iid)
            for pos, val in zip(positions, values):
                new_iid[pos] = val
            yield "".join(_HEX[n] for n in prefix + new_iid)


# ---------------------------------------------------------------------------
# Seed loader
# ---------------------------------------------------------------------------

def _load_seeds(path: str) -> list[str]:
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
