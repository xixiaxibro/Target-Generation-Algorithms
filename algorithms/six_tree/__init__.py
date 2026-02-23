"""
6tree: Tree-based IPv6 Target Generation Algorithm.

Ported to Python from the original C++ implementation by Zhizhu Liu (2019).
Reference: https://github.com/tumi8/tma-23-target-generation/tree/main/algorithms/6tree

Algorithm
---------
6tree discovers active IPv6 addresses by exploiting the non-uniform
distribution of the known-active address space:

  1. **Translate** — Seed addresses are normalised to b4 format (32
     lowercase hex chars, no colons), which treats each address as a
     32-dimensional vector over the alphabet {0..f}.

  2. **DHC tree** — A Dimensional Hierarchical Clustering tree is built
     by recursively partitioning the seed set on the leftmost nibble
     dimension that still has variation.  Each leaf node represents a
     small, uniform sub-region of the address space.

  3. **Generate** — Leaf nodes are ranked by seed density
     (seed_count / region_size).  Wildcard nibbles in each leaf's
     pattern are filled: exhaustively for small regions, randomly for
     large ones.  This biases candidate generation toward the densest
     parts of the address space.

Usage
-----
    from algorithms.six_tree import run
    run(seeds="seeds.txt", output="targets.txt", budget=1_000_000)

Or via the top-level CLI:
    python main.py --algorithm 6tree --seeds seeds.txt --output targets.txt
"""

from __future__ import annotations

from .generation import generate_targets
from .translation import normalize_to_b4
from .tree import build_dhc_tree


def run(seeds: str, output: str, budget: int) -> None:
    """
    Generate IPv6 target addresses using the 6tree algorithm.

    Args:
        seeds:  Path to seed IPv6 address file (one address per line).
                Addresses may be in any format supported by the translation
                module: standard (with or without :: compression), b1 (128
                binary chars), b2 (64 base-4 chars), b3 (42 octal chars),
                b4 (32 hex chars), or b5 (25 base-32 chars).
        output: Path to write the generated candidate IPv6 addresses.
                Each line contains one address in full standard notation
                (XXXX:XXXX:XXXX:XXXX:XXXX:XXXX:XXXX:XXXX).
        budget: Maximum number of addresses to generate.
    """
    # Step 1: load and normalise seeds
    print(f"[6tree] Loading seeds from: {seeds}")
    b4_seeds = _load_seeds(seeds)
    if not b4_seeds:
        print("[6tree] ERROR: no valid seed addresses found — aborting.")
        return
    print(f"[6tree] {len(b4_seeds):,} unique seed addresses loaded.")

    # Step 2: build DHC space tree
    print("[6tree] Building DHC space tree ...")
    root = build_dhc_tree(b4_seeds)
    stats = root.stats()
    print(
        f"[6tree] Tree built: {stats['total_nodes']:,} nodes, "
        f"{stats['leaf_nodes']:,} leaves, max depth {stats['max_depth']}."
    )

    # Step 3: generate targets
    print(f"[6tree] Generating up to {budget:,} target addresses ...")
    targets = generate_targets(root, budget)
    print(f"[6tree] {len(targets):,} target addresses generated.")

    # Step 4: write output
    with open(output, "w") as fh:
        for addr in targets:
            fh.write(addr + "\n")
    print(f"[6tree] Targets written to: {output}")


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
