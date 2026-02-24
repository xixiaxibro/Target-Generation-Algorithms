"""
entropy-ip: Entropy-based IPv6 target generation.

Paper: "Entropy/IP: Uncovering Structure in IPv6 Addresses"
       Foremski et al., IMC 2016 (Akamai Technologies)
Reference: https://github.com/tumi8/tma-23-target-generation/tree/main/algorithms

Algorithm
---------
1. **Load** — seeds are normalised to b4 (32 lowercase hex chars) and
   converted to a numpy uint8 nibble matrix (n, 32).

2. **Segment** — per-nibble Shannon entropy (log2) is computed; boundaries
   are placed at nibble 8 (ISP prefix boundary), nibble 16 (network ID
   boundary), and wherever the entropy tier changes in the interface portion
   (nibbles 16–31).  Faithfully replicates a1-segments.py.

3. **Mine** — for each segment, three passes find structural patterns:
   (a) heavy-hitter const values via IQR outlier detection,
   (b) dense clusters via DBSCAN (density ≥ 100× uniform expectation),
   (c) contiguous value ranges via gap-based grouping.
   Faithfully replicates a2-mining.py.

4. **Generate** — each target address is produced by independently sampling
   every segment from its pattern distribution (const → exact value;
   range → uniform draw).  This replaces the Bayesian network
   (a4-bayes.sh + c1-gen.py) with independent segment sampling — a valid
   simplification when inter-segment correlations are weak.

Usage
-----
    from algorithms.entropy_ip import run
    run(seeds="seeds.txt", output="targets.txt", budget=1_000_000)
"""

from __future__ import annotations

import numpy as np

from algorithms.six_tree.translation import normalize_to_b4
from .generation import generate_addresses
from .mining import mine_segment, seg_to_ints
from .segments import find_segments


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(seeds: str, output: str, budget: int) -> None:
    """Generate IPv6 target addresses using the entropy-ip algorithm.

    Args:
        seeds:  Path to seed IPv6 address file (one per line).
        output: Path to write generated target addresses.
        budget: Maximum number of addresses to generate.
    """
    # ── Step 1: load and normalise seeds ─────────────────────────────────────
    print(f"[entropy-ip] Loading seeds from: {seeds}")
    b4_seeds = _load_seeds(seeds)
    if not b4_seeds:
        print("[entropy-ip] ERROR: no valid seed addresses — aborting.")
        return
    print(f"[entropy-ip] {len(b4_seeds):,} unique seeds loaded.")

    # Convert b4 strings → nibble matrix (n, 32) uint8
    arrs = np.array(
        [[int(c, 16) for c in addr] for addr in b4_seeds], dtype=np.uint8
    )

    # ── Step 2: entropy segmentation ─────────────────────────────────────────
    print("[entropy-ip] Computing entropy segments …")
    segments = find_segments(arrs)
    print(
        f"[entropy-ip] {len(segments)} segments: "
        + ", ".join(f"[{s['start']},{s['stop']})" for s in segments)
    )

    # ── Step 3: mine patterns per segment ─────────────────────────────────────
    print("[entropy-ip] Mining segment patterns …")
    seg_patterns: list[list[dict]] = []
    for seg in segments:
        start, stop = seg["start"], seg["stop"]
        width = stop - start
        seg_bits = width * 4
        vals = seg_to_ints(arrs, start, stop)
        patterns = mine_segment(vals, seg_bits)
        seg_patterns.append(patterns)
        n_const = sum(1 for p in patterns if p["type"] == "const")
        n_range = sum(1 for p in patterns if p["type"] == "range")
        print(f"  [{start},{stop}): {n_const} const + {n_range} range patterns")

    # ── Step 4: generate target addresses ────────────────────────────────────
    print(f"[entropy-ip] Generating up to {budget:,} target addresses …")
    targets = generate_addresses(arrs, segments, seg_patterns, budget)
    print(f"[entropy-ip] {len(targets):,} addresses generated.")

    # ── Write output ──────────────────────────────────────────────────────────
    with open(output, "w") as fh:
        for addr in targets:
            fh.write(addr + "\n")
    print(f"[entropy-ip] Targets written to: {output}")


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
