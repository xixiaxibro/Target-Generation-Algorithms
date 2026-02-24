# Algorithm Critique: Structural Issues in the TGA Implementation

> **Purpose**: Document structural limitations found during implementation of the five
> offline TGA algorithms (6tree, 6Forest, 6Graph, DET, entropy-ip).  No code is changed
> here; this file is a reference for future redesign decisions.

---

## Table of Contents

1. [P1 — DHC Generation Layer: Independence Assumption](#p1--dhc-generation-layer-independence-assumption)
2. [P1 — entropy-ip: Independent Segment Sampling Breaks Under Mixed ISPs](#p1--entropy-ip-independent-segment-sampling-breaks-under-mixed-isps)
3. [P2 — Leaf Stop Condition Is Seed-Count-Based, Not Density-Based](#p2--leaf-stop-condition-is-seed-count-based-not-density-based)
4. [P2 — 6Graph Density Gate Is Effectively a No-Op](#p2--6graph-density-gate-is-effectively-a-no-op)
5. [P3 — Split Heuristic Quality Ordering and Unit Mismatch](#p3--split-heuristic-quality-ordering-and-unit-mismatch)
6. [P4 — DET Lacks Outlier Filtering](#p4--det-lacks-outlier-filtering)
7. [Design Assumptions for a Future Redesign](#design-assumptions-for-a-future-redesign)

---

## P1 — DHC Generation Layer: Independence Assumption

**Affects**: 6tree, 6Forest, 6Graph, DET (all four share the same generation logic)

**File**: `algorithms/six_forest/generation.py` (re-exported by 6Graph and DET)

### What the code does

`region_to_pattern()` (line 33) condenses a leaf region into a wildcard pattern:

```python
for dim in range(32):
    unique = np.unique(arrs[:, dim])
    pattern.append(_HEX[unique[0]] if len(unique) == 1 else "*")
```

A dimension gets `*` if *any* two seeds differ there, regardless of how many distinct
values exist.  The expansion step (`_expand_plain`, line 50) then fills every wildcard
position with all 16 nibbles independently via `itertools.product`:

```python
for combo in itertools.product(_HEX, repeat=len(wildcards)):
```

### Why this is wrong

Consider a leaf with two seeds:

```
seed A:  2001:0db8:0001:0001:...
seed B:  2001:0db8:0002:0002:...
```

The pattern becomes `2001:0db8:***:***:...`.  The cross-product generates
`16 × 16 = 256` combinations, including `2001:0db8:0001:0002:...` and
`2001:0db8:0002:0001:...` — addresses that combine nibbles from *different seeds* in
ways that no seed supports.

In practice IPv6 subnets are hierarchical: nibble positions in the *same dimension
group* are correlated (e.g., nibbles 8–15 encode the network prefix, nibbles 16–31
encode the interface identifier within that prefix).  The generation layer destroys
this correlation by treating every wildcard dimension as independent, producing
*diagonal* combinations that exist in the mathematical Cartesian product but not in
any real routing block.

The per-dimension fallback (`_expand_distance`, line 65) is better — it cycles around
each individual seed — but it still mixes nibble values across seeds at each wildcard
position, so the fundamental issue remains for regions with ≥ 2 seeds.

### Quantitative impact

For a leaf with `k` wildcard dimensions, the cross-product generates up to `16^k`
candidates from a region that may contain only `O(1)` actual routing blocks.  With
`k = 6` (typical for a /48–/64 expansion), that is 16.7 million addresses per leaf.

---

## P1 — entropy-ip: Independent Segment Sampling Breaks Under Mixed ISPs

**Affects**: entropy-ip only

**File**: `algorithms/entropy_ip/generation.py`, function `_sample_segment` (line 38)

### What the code does

Each segment is sampled independently:

```python
for seg, patterns in zip(segments, seg_patterns):
    val = _sample_segment(patterns, width)
```

The docstring explicitly acknowledges the assumption:

> "This is a valid simplification when inter-segment correlations are weak — each
> segment is sampled from its mined pattern distribution without conditioning on
> other segments."

### Why this is wrong

In real-world IPv6 address space, routing prefix bytes (nibbles 0–15) are strongly
correlated with interface identifier bytes (nibbles 16–31) because each ISP or
organisation uses a characteristic IID assignment policy for its own prefixes.  A seed
set drawn from multiple ISPs will exhibit multi-modal segment distributions:

- ISP A assigns EUI-64 IIDs under its prefixes.
- ISP B assigns random/privacy IIDs under its prefixes.

Independent sampling draws a prefix at random from the combined distribution, then
draws an IID at random from the combined distribution.  This generates addresses that
pair ISP-A prefixes with ISP-B IID patterns — combinations that correspond to no
real host.

The original entropy-ip used a Bayesian network (`a4-bayes.sh`) precisely to model
segment dependencies.  Replacing it with independent sampling is only valid for
single-ISP or very homogeneous seed sets.

---

## P2 — Leaf Stop Condition Is Seed-Count-Based, Not Density-Based

**Affects**: 6Forest, 6Graph, DET (all use `min_region_size = 16` as the leaf threshold)

**Files**:
- `algorithms/six_forest/partition.py:80` — `dhc_forest(min_region_size=16)`
- `algorithms/det/partition.py:72` — `dhc_det(min_region_size=16)`
- `algorithms/six_graph/partition.py` (BFS variant, same threshold)

### What the code does

DHC stops splitting a region when it contains ≤ `min_region_size` addresses:

```python
if len(region) <= min_region_size:
    regions.append(region)
    continue
```

### Why this is wrong

16 seeds spread over a /32 allocation (2³² addresses each) is an extremely sparse
region; 16 seeds packed into a /64 subnet (65 536 addresses) is a dense region.
The DHC treats both identically: both become leaves, and both are expanded via the
same wildcard pattern logic.

The leaf criterion should account for the *spatial density* of seeds within the
region's address space, not just the absolute seed count.  A sparse leaf with 16
seeds over a /32 generates millions of unlikely targets; a dense leaf with 16 seeds
over a /64 generates a tight, high-probability target list.

Without a density-aware stop condition, the budget allocated to sparse leaves is
wasted on improbable targets.

---

## P2 — 6Graph Density Gate Is Effectively a No-Op

**Affects**: 6Graph only

**File**: `algorithms/six_graph/graph.py`, functions `_density` (line 40) and
`cluster_by_graph` (line 134)

### What the code does

The graph clustering rejects an edge if the density of either endpoint's neighbourhood
falls below `min_density` (default `0.8`):

```python
if dens_i < min_density or dens_j < min_density:
    G.remove_edge(i, j)
```

The density formula is:

```python
density = Σ_d |unique(arrs[:, d])| / (n * 32)
```

### Why the threshold is unreachable

Consider the smallest possible neighbourhood: two addresses (n = 2) that differ in
exactly one nibble position (Hamming distance = 1, the tightest possible pair):

```
distinct_count = 31 × 1 + 1 × 2 = 33
density        = 33 / (2 × 32)  = 0.516
```

`0.516 < 0.8` → the edge is **rejected**.

For any n ≥ 2:

```
density ≤ (32 × min(n, 16)) / (n × 32) = min(n, 16) / n ≤ 1
```

To reach density ≥ 0.8 with n = 2, we need `Σ |unique(d)| ≥ 0.8 × 2 × 32 = 51.2`.
The maximum is `32 × 2 = 64` (all 32 dimensions differ), so we would need ≥ 52 of 32
dimensions to differ — impossible.

**Consequence**: The density gate fires on every edge for every neighbourhood of size
≥ 2.  No edges are ever accepted.  All addresses become isolated nodes (outliers), and
the graph clustering step is a no-op.  6Graph then runs its three outlier-iteration
passes on the full seed set, effectively degenerating to a modified 6Forest without
the IsolationForest filter.

### Root cause

The formula divides by `n`, making it size-dependent rather than scale-invariant.  The
threshold `0.8` was likely calibrated against a different formula (e.g., fraction of
non-wildcard dimensions, or address-space utilisation).

---

## P3 — Split Heuristic Quality Ordering and Unit Mismatch

**Affects**: 6tree (leftmost), 6Forest (tiebreak comparison)

### 6tree: Leftmost Split Is the Weakest Heuristic

**File**: `algorithms/six_tree/tree.py:149–156`

```python
for dim in range(32):
    if pattern[dim] != "*":
        continue
    values = {addr[dim] for addr in addresses}
    if len(values) > 1:
        split_dim = dim
        break
```

6tree always splits on the **first** wildcard dimension that has more than one value.
This is equivalent to building a radix trie and is the weakest possible heuristic:

- It ignores the distribution of values (a 15:1 split is treated identically to an 8:8
  split).
- It ignores downstream dimensions (splitting leftmost may leave all variance in
  rightmost dimensions, requiring deeper trees).

Both 6Forest (maxcovering) and DET (min-entropy) are strictly stronger heuristics
because they consider all dimensions before deciding.

### 6Forest: Tiebreak Compares Incompatible Units

**File**: `algorithms/six_forest/partition.py:67`

```python
if best_cov - leftmost_cov <= best_idx - leftmost_idx:
    best_idx = leftmost_idx
```

- `best_cov - leftmost_cov`: difference in maxcovering score (units: **address count**)
- `best_idx - leftmost_idx`: difference in dimension index (units: **nibble position**)

These are compared directly, implying that 1 address of covering gain equals 1 nibble
of positional shift.  For a region with 10 000 seeds, covering differences are
O(thousands); positional shifts are at most 31.  The tiebreak almost always favours
the better-covering dimension regardless of position, making the "prefer leftmost"
intent ineffective for large regions.  Conversely, for tiny regions (n ≈ 16 = leaf
threshold), covering differences are O(1–16), and the tiebreak can erroneously prefer
a far-right dimension over a left one with only marginally lower score.

---

## P4 — DET Lacks Outlier Filtering

**Affects**: DET only

**File**: `algorithms/det/partition.py` — no outlier step exists

### What is missing

6Forest applies `algorithms/six_forest/outliers.py` (IsolationForest + Four-Deviations
removal) before generating targets.  This step identifies seeds that are structurally
anomalous (far from any cluster centroid) and either down-weights them or removes them
before expansion.

DET has no equivalent.  Every seed, including genuine outliers, becomes the centre of
a leaf region and contributes targets.

### Why this matters

The original DET algorithm is **online**: scanning feedback prunes low-yield regions
mid-run.  Outlier leaves that generate no hits are abandoned quickly.

This implementation is **offline**: all leaf regions contribute to the final target
list up-front.  Without an offline outlier filter, the budget is diluted by targets
around anomalous seeds that would have been pruned immediately in the online setting.

---

## Design Assumptions for a Future Redesign

The following assumptions underlie the five algorithms.  A new implementation should
treat each as a hypothesis to validate rather than a given:

| # | Assumption | Held by | Evidence for | Evidence against |
|---|------------|---------|-------------|-----------------|
| A1 | Nibble dimensions are independent within a DHC leaf | All DHC algorithms | Simplifies enumeration | IPv6 address structure has hierarchical dependencies between dimension groups |
| A2 | Segment values are independent across segments | entropy-ip | Works for single-ISP data | Multi-ISP seeds create multi-modal joint distributions |
| A3 | Seed count is a proxy for spatial density | All DHC algorithms (leaf threshold) | Easy to implement | Sparse vs dense regions have identical leaf criteria |
| A4 | The 6Graph density formula is scale-invariant | 6Graph | — | Formula divides by n; threshold 0.8 is unreachable for n ≥ 2 |
| A5 | Leftmost split is sufficient | 6tree | O(1) per split | Ignores value distribution; weaker than maxcovering and min-entropy |
| A6 | Outlier seeds self-prune under budget pressure | DET | True for online scanning | Not true offline; outliers consume budget proportional to their leaf count |

### If redesigning from scratch

1. **Replace pattern-level independence** with a model that preserves intra-leaf
   correlations, e.g., sample from the actual joint distribution of seed nibbles within
   each leaf rather than the cross-product of marginals.
2. **Replace segment independence** with at minimum a Gaussian mixture model per
   segment conditioned on a routing-prefix cluster label, or restore the original
   Bayesian network.
3. **Replace count-based leaf stopping** with a density-based criterion: compute the
   empirical density (seeds / address-space-size of the region) and stop when it
   exceeds a threshold.
4. **Fix or remove the 6Graph density gate**: either use a scale-invariant formula
   (e.g., mean pairwise Hamming distance / 32) or lower the threshold to a reachable
   value (≤ 0.1 for typical cluster sizes).
5. **Add offline outlier detection to DET**: the Four-Deviations method from 6Forest
   is a lightweight drop-in.

---

*Generated: 2026-02-24 — covers algorithms as of commit `15739bb`.*
