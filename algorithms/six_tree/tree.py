"""
DHC (Dimensional Hierarchical Clustering) space tree.

Ported from 6tree function2_G (C++ by Zhizhu Liu, 2019).

The tree recursively partitions a set of b4-format IPv6 addresses by
nibble dimension (0–31, left-to-right).  At each node the algorithm
picks the leftmost nibble position that still has more than one unique
value among the current address set and branches on every value found
there.  Leaf nodes represent uniform sub-regions (or regions with only
one address).

Each node stores:
  pattern     32-char string where fixed nibbles hold their hex digit
              and unresolved nibbles hold '*'.
  addresses   The seed addresses that fall in this node's region.
  children    Mapping from nibble char → child SpaceTreeNode.
  split_dim   Nibble index used to create children (-1 for leaves).
  depth       Distance from root.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterator, List


@dataclass
class SpaceTreeNode:
    pattern: str                         # 32-char pattern, '*' = wildcard
    addresses: List[str]                 # b4 seeds in this region
    children: Dict[str, SpaceTreeNode]   # nibble char → child
    split_dim: int = -1                  # nibble index used to split (-1 = leaf)
    depth: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def seed_count(self) -> int:
        return len(self.addresses)

    @property
    def wildcard_count(self) -> int:
        return self.pattern.count("*")

    @property
    def region_size(self) -> int:
        """Total addresses in this region: 16 ** (number of wildcards)."""
        wc = self.wildcard_count
        # Cap at a large int to avoid memory issues; Python handles big ints fine
        return 16 ** wc

    @property
    def density(self) -> float:
        """
        Seed density = seed_count / region_size.

        Higher density → more seeds per addressable slot → higher scan priority.
        """
        rs = self.region_size
        return self.seed_count / rs if rs > 0 else 0.0

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def leaves(self) -> Iterator[SpaceTreeNode]:
        """BFS iterator over all leaf nodes."""
        queue: deque[SpaceTreeNode] = deque([self])
        while queue:
            node = queue.popleft()
            if node.is_leaf:
                yield node
            else:
                queue.extend(node.children.values())

    def all_nodes(self) -> Iterator[SpaceTreeNode]:
        """BFS iterator over every node (root first)."""
        queue: deque[SpaceTreeNode] = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children.values())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary dict (useful for debugging)."""
        nodes = list(self.all_nodes())
        leaves = [n for n in nodes if n.is_leaf]
        return {
            "total_nodes": len(nodes),
            "leaf_nodes": len(leaves),
            "max_depth": max(n.depth for n in nodes),
            "seed_count": self.seed_count,
        }


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_dhc_tree(addresses: List[str], max_depth: int = 32) -> SpaceTreeNode:
    """
    Build a DHC space tree from b4-format seed addresses.

    Args:
        addresses:  List of 32-char lowercase hex strings.
        max_depth:  Maximum nibble depth (default 32 covers all nibbles).

    Returns:
        Root SpaceTreeNode.
    """
    unique = sorted(set(addresses))
    return _dhc("*" * 32, unique, depth=0, max_depth=max_depth)


# ---------------------------------------------------------------------------
# Internal recursive builder
# ---------------------------------------------------------------------------

def _dhc(
    pattern: str,
    addresses: List[str],
    depth: int,
    max_depth: int,
) -> SpaceTreeNode:
    """Recursive DHC partitioning step."""
    # Base cases
    if not addresses or depth >= max_depth:
        return SpaceTreeNode(
            pattern=pattern,
            addresses=addresses,
            children={},
            split_dim=-1,
            depth=depth,
        )

    # Find the leftmost wildcard dimension with more than one unique value
    split_dim = -1
    for dim in range(32):
        if pattern[dim] != "*":
            continue  # already fixed at a higher level
        values = {addr[dim] for addr in addresses}
        if len(values) > 1:
            split_dim = dim
            break

    if split_dim == -1:
        # Every wildcard dimension is uniform → leaf
        return SpaceTreeNode(
            pattern=pattern,
            addresses=addresses,
            children={},
            split_dim=-1,
            depth=depth,
        )

    # Partition addresses by their nibble at split_dim
    groups: Dict[str, List[str]] = {}
    for addr in addresses:
        c = addr[split_dim]
        if c not in groups:
            groups[c] = []
        groups[c].append(addr)

    children: Dict[str, SpaceTreeNode] = {}
    for val in sorted(groups):
        child_pattern = pattern[:split_dim] + val + pattern[split_dim + 1 :]
        children[val] = _dhc(
            child_pattern, groups[val], depth + 1, max_depth
        )

    return SpaceTreeNode(
        pattern=pattern,
        addresses=addresses,
        children=children,
        split_dim=split_dim,
        depth=depth,
    )
