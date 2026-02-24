"""
Target address generation from 6Forest patterns.

Two expansion strategies (matching 6forest_to_targetlist.py)
------------------------------------------------------------
≤ max_wildcards_plain wildcards  →  full cross-product (exhaustive).
>  max_wildcards_plain wildcards  →  per-dimension expansion: for every seed
    address in the region, cycle each wildcard position independently through
    0-f.  This avoids enormous enumerations (e.g. 16^10 ≈ 1 trillion) while
    still sampling around each known seed.

Patterns are sorted by descending density before expansion so that the most
seed-dense regions are generated first — consistent with 6tree's priority.
"""

from __future__ import annotations

import itertools
from typing import Iterator, List, Set

import numpy as np

from algorithms.six_tree.translation import b4_to_std

_HEX: str = "0123456789abcdef"
_MAX_PLAIN: int = 3          # wildcards threshold for cross-product vs. distance


# ---------------------------------------------------------------------------
# Pattern extraction
# ---------------------------------------------------------------------------

def region_to_pattern(arrs: np.ndarray) -> str:
    """Derive the wildcard pattern string from a region array.

    Dimensions with a single unique nibble value → that hex digit.
    Dimensions with multiple values → '*'.
    """
    pattern: List[str] = []
    for dim in range(32):
        unique = np.unique(arrs[:, dim])
        pattern.append(_HEX[unique[0]] if len(unique) == 1 else "*")
    return "".join(pattern)


# ---------------------------------------------------------------------------
# Expansion strategies
# ---------------------------------------------------------------------------

def _expand_plain(
    pattern: str,
    wildcards: List[int],
) -> Iterator[str]:
    """Full cross-product enumeration over all wildcard positions."""
    if not wildcards:
        yield pattern
        return
    chars = list(pattern)
    for combo in itertools.product(_HEX, repeat=len(wildcards)):
        for pos, ch in zip(wildcards, combo):
            chars[pos] = ch
        yield "".join(chars)


def _expand_distance(
    pattern: str,
    arrs: np.ndarray,
    wildcards: List[int],
) -> Iterator[str]:
    """Per-dimension expansion around each seed address in the region.

    For each seed, cycle every wildcard position independently through 0-f.
    Yields only unique addresses.
    """
    seen: Set[str] = set()
    for row in arrs:
        seed = "".join(_HEX[n] for n in row)
        for wpos in wildcards:
            for ch in _HEX:
                addr = seed[:wpos] + ch + seed[wpos + 1:]
                if addr not in seen:
                    seen.add(addr)
                    yield addr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_from_patterns(
    pattern_regions: List[np.ndarray],
    budget: int,
    max_wildcards_plain: int = _MAX_PLAIN,
) -> List[str]:
    """Generate up to *budget* IPv6 target addresses from pattern regions.

    Args:
        pattern_regions:      List of (k, 32) uint8 arrays, each a leaf region.
        budget:               Maximum number of output addresses.
        max_wildcards_plain:  Regions with this many or fewer wildcards are
                              expanded exhaustively; larger regions use the
                              per-dimension strategy.

    Returns:
        List of standard IPv6 address strings, deduplicated, budget-limited.
    """
    if not pattern_regions or budget <= 0:
        return []

    def _density(arrs: np.ndarray) -> float:
        n_wc = sum(
            1 for d in range(32) if len(np.unique(arrs[:, d])) > 1
        )
        region_size = 16 ** n_wc
        return len(arrs) / region_size if region_size > 0 else 0.0

    # Highest density first.
    sorted_regions = sorted(pattern_regions, key=_density, reverse=True)

    results: List[str] = []
    seen: Set[str] = set()

    for arrs in sorted_regions:
        if len(results) >= budget:
            break

        pattern  = region_to_pattern(arrs)
        wildcards = [i for i, c in enumerate(pattern) if c == "*"]

        if len(wildcards) <= max_wildcards_plain:
            it: Iterator[str] = _expand_plain(pattern, wildcards)
        else:
            it = _expand_distance(pattern, arrs, wildcards)

        for b4 in it:
            if b4 not in seen:
                seen.add(b4)
                results.append(b4_to_std(b4))
                if len(results) >= budget:
                    break

    return results
