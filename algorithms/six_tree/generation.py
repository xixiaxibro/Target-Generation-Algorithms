"""
Target address generation from the 6tree DHC space tree.

Corresponds to 6tree function3_L (local simulation) and the offline part
of function4_R (real search without network feedback).

Generation strategy
-------------------
1.  Collect all leaf nodes from the DHC tree.
2.  Sort leaves by descending density (seed_count / region_size) so that
    the most densely seeded address regions are expanded first — this is
    the core insight of 6tree: dense regions in the seed set are more
    likely to contain undiscovered active addresses.
3.  For each leaf node expand its wildcard pattern:
      * If the region has ≤ max_count remaining slots, enumerate every
        address in the region (exhaustive fill).
      * Otherwise, sample randomly from the region.
4.  Deduplicate and stop when `budget` addresses have been collected.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Iterator, List, Set

from .translation import b4_to_std
from .tree import SpaceTreeNode

_HEX_CHARS: str = "0123456789abcdef"

# Threshold: if a region has more than this many addresses we sample rather
# than enumerate (avoids enormous enumerations for low-density leaves).
_MAX_ENUM_SIZE: int = 16 ** 6   # 16 million — enum anything smaller


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_targets(root: SpaceTreeNode, budget: int) -> List[str]:
    """
    Generate up to `budget` IPv6 target addresses from the DHC tree.

    Args:
        root:   Root of the DHC space tree built from seed addresses.
        budget: Maximum number of addresses to produce.

    Returns:
        List of standard IPv6 address strings (XXXX:XXXX:...).
    """
    if budget <= 0:
        return []

    # Collect leaves and rank by density (highest first)
    leaves: List[SpaceTreeNode] = sorted(
        root.leaves(), key=lambda n: -n.density
    )

    results: List[str] = []
    seen: Set[str] = set()

    for node in leaves:
        if len(results) >= budget:
            break
        remaining = budget - len(results)
        for b4 in _expand_pattern(node.pattern, remaining):
            if b4 not in seen:
                seen.add(b4)
                results.append(b4_to_std(b4))
                if len(results) >= budget:
                    break

    return results


# ---------------------------------------------------------------------------
# Pattern expansion
# ---------------------------------------------------------------------------

def _expand_pattern(pattern: str, max_count: int) -> Iterator[str]:
    """
    Yield up to max_count b4 addresses matching `pattern`.

    '*' positions are filled with hex digits [0-9a-f].
    Fixed positions are kept as-is.
    """
    wildcards = [i for i, c in enumerate(pattern) if c == "*"]
    n_wc = len(wildcards)

    if n_wc == 0:
        yield pattern
        return

    region_size = 16 ** n_wc

    if region_size <= max_count and region_size <= _MAX_ENUM_SIZE:
        # Full enumeration
        yield from _enumerate_pattern(list(pattern), wildcards, 0)
    else:
        # Random sampling without replacement
        yield from _sample_pattern(pattern, wildcards, min(max_count, region_size))


def _enumerate_pattern(
    addr: List[str],
    wildcards: List[int],
    idx: int,
) -> Iterator[str]:
    """Depth-first enumeration of all addresses matching a wildcard pattern."""
    if idx == len(wildcards):
        yield "".join(addr)
        return
    pos = wildcards[idx]
    for c in _HEX_CHARS:
        addr[pos] = c
        yield from _enumerate_pattern(addr, wildcards, idx + 1)


def _sample_pattern(
    pattern: str,
    wildcards: List[int],
    count: int,
) -> Iterator[str]:
    """
    Randomly sample `count` unique addresses from a wildcard pattern.

    Uses a rejection-sampling loop with an early-exit guard so we don't
    spin forever on nearly-full regions.
    """
    seen: Set[str] = set()
    max_attempts = count * 8  # allow up to 8x rejections before giving up
    attempts = 0

    while len(seen) < count and attempts < max_attempts:
        addr = list(pattern)
        for i in wildcards:
            addr[i] = random.choice(_HEX_CHARS)
        b4 = "".join(addr)
        if b4 not in seen:
            seen.add(b4)
            yield b4
        attempts += 1
