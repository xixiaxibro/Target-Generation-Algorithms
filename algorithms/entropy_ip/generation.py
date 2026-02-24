"""Address generation via independent per-segment sampling (entropy-ip).

Replaces the Bayesian network (a4-bayes.sh + c1-gen.py) from the original
entropy-ip pipeline with independent segment sampling.  This is a valid
simplification when inter-segment correlations are weak — each segment is
sampled from its mined pattern distribution without conditioning on other
segments.

Sampling procedure
------------------
For each candidate address:
  1. For every segment, choose one pattern by weighted random selection.
  2. If the pattern is "const"  → use the stored value directly.
     If the pattern is "range"  → draw uniformly from [lo, hi].
  3. Convert the segment integer to nibbles (big-endian within segment).
  4. Concatenate all segments → 32 hex chars → standard IPv6 notation.

Deduplication is enforced via a seen-set; the loop runs until *budget*
unique addresses are collected or max_attempts = budget * 10 is reached.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np

from algorithms.six_tree.translation import b4_to_std

_HEX: str = "0123456789abcdef"


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_segment(patterns: list[dict], seg_width: int) -> int:
    """Randomly draw one integer value for a segment.

    Args:
        patterns:  List of pattern dicts (from mine_segment).
        seg_width: Width of the segment in nibbles.

    Returns:
        Integer in [0, 16**seg_width - 1].
    """
    if not patterns:
        return random.randint(0, 16 ** seg_width - 1)

    weights = [p["weight"] for p in patterns]
    total = sum(weights)
    if total <= 0:
        return random.randint(0, 16 ** seg_width - 1)

    r = random.random() * total
    cumul = 0.0
    for p in weights:
        cumul += p
        if r <= cumul:
            break
    else:
        # Floating-point edge: pick last
        pass

    # Find which pattern was selected
    cumul = 0.0
    for p in patterns:
        cumul += p["weight"]
        if r <= cumul:
            if p["type"] == "const":
                return p["value"]
            return random.randint(p["lo"], p["hi"])

    # Fallback
    p = patterns[-1]
    return p["value"] if p["type"] == "const" else random.randint(p["lo"], p["hi"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_addresses(
    arrs: np.ndarray,
    segments: list[dict],
    seg_patterns: list[list[dict]],
    budget: int,
) -> List[str]:
    """Generate *budget* IPv6 addresses by independently sampling each segment.

    Args:
        arrs:         Seed nibble matrix (n, 32) uint8 — used only for
                      segment width resolution (could also be inferred from
                      segments, but kept for consistency with the pipeline).
        segments:     Segment descriptors from find_segments().
        seg_patterns: Per-segment pattern lists from mine_segment(), in the
                      same order as *segments*.
        budget:       Maximum number of distinct addresses to generate.

    Returns:
        Deduplicated list of standard IPv6 address strings (≤ budget items).
    """
    results: List[str] = []
    seen: set[str] = set()
    max_attempts = budget * 10

    for attempt in range(max_attempts):
        if len(results) >= budget:
            break

        nibbles: list[int] = []
        for seg, patterns in zip(segments, seg_patterns):
            start, stop = seg["start"], seg["stop"]
            width = stop - start
            val = _sample_segment(patterns, width)

            # Integer → nibbles (big-endian within this segment)
            seg_nibs: list[int] = []
            for _i in range(width):
                seg_nibs.append(val & 0xF)
                val >>= 4
            seg_nibs.reverse()
            nibbles.extend(seg_nibs)

        if len(nibbles) != 32:
            continue  # guard against misconfigured segments

        b4 = "".join(_HEX[nib] for nib in nibbles)
        if b4 not in seen:
            seen.add(b4)
            results.append(b4_to_std(b4))

    return results
