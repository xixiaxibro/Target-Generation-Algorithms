"""
Target address generation for 6Graph.

The generation logic is identical to 6Forest — pattern wildcards are sorted
by descending seed density and expanded exhaustively (≤3 wildcards) or via
per-dimension cycling (>3 wildcards).

This module simply re-exports the two public symbols from six_forest.generation
so that 6Graph callers have a stable local import path.
"""

from algorithms.six_forest.generation import region_to_pattern, generate_from_patterns  # noqa: F401
