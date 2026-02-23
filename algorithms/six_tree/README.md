# 6tree — Python Port

Python re-implementation of the **6tree** IPv6 target generation algorithm.

Original C++ source: [tumi8/tma-23-target-generation/algorithms/6tree](https://github.com/tumi8/tma-23-target-generation/tree/main/algorithms/6tree)
Original author: Zhizhu Liu (2019)

---

## Algorithm

6tree treats each IPv6 address as a **32-dimensional vector** (one nibble per dimension in b4/hex format) and builds a **Dimensional Hierarchical Clustering (DHC) tree** from a set of seed addresses.

### Steps

1. **Translate** — Seed addresses are normalised to *b4* format: 32 lowercase hex characters with no colons (e.g. `20010db885a300000000 8a2e03707334`). The `ipaddress` standard library handles all compressed/expanded IPv6 notations automatically.

2. **Build DHC tree** — Starting from a root node whose pattern is `********************************` (all wildcards), the algorithm recursively partitions the seed set:
   - Find the leftmost nibble position (0–31) where the current address set has more than one unique value.
   - Branch on each distinct value at that position (up to 16 children).
   - Recurse on each branch with the corresponding sub-set of seeds.
   - Stop when a node's addresses all agree on every remaining wildcard position (leaf), or when all 32 nibbles are fixed.

3. **Generate targets** — Leaf nodes are ranked by **density** (`seed_count / 16^wildcards`). For each leaf (highest density first), wildcard nibble positions are filled:
   - **Exhaustive fill** if the region has ≤ 16 million addresses.
   - **Random sampling** otherwise.
   - Duplicate addresses are discarded.

### Why this works

Dense seed regions in the IPv6 address space are empirically more likely to contain undiscovered active hosts. By biasing generation toward those regions, 6tree achieves a much higher hit-rate than uniform random scanning.

---

## File Structure

```
six_tree/
├── __init__.py       run() entry point, _load_seeds()
├── translation.py    IPv6 format conversions (std / b1 / b2 / b3 / b4 / b5)
├── tree.py           DHC space tree (SpaceTreeNode, build_dhc_tree)
├── generation.py     Target generation from the tree
└── README.md         This file
```

---

## Supported Seed Formats

| Format | Length | Description |
|--------|--------|-------------|
| std    | varies | Standard IPv6 (`2001:db8::1`, `::1`, etc.) |
| b4     | 32     | Hex nibbles, no colons |
| b1     | 128    | Binary digits |
| b2     | 64     | Base-4 digits |
| b3     | 42     | Octal digits (top 2 bits dropped) |
| b5     | 25     | Base-32 digits (top 3 bits dropped — lossy) |

---

## Differences from Original C++

| Aspect | Original C++ | This Python port |
|--------|-------------|-----------------|
| Network scanning | Calls ZMapv6 for live probing | Not applicable — offline generation only |
| Alias detection | Full ADET pipeline | Not implemented (requires live network) |
| Feedback loop | Adaptive scan based on responses | Not applicable offline |
| Formats | All 6 formats | All 6 formats |
| DHC split heuristic | Leftmost variable dimension | Same |
| Target priority | Node density | Same |

The core address-generation logic is faithful to the original. The network-scanning and alias-detection components are omitted because they require live probing infrastructure (ZMapv6).
