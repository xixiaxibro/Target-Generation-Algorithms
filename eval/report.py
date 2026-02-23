"""
Metrics report formatting and persistence.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def print_report(metrics: Dict[str, Any]) -> None:
    """Print a formatted metrics report to stdout."""
    algo    = metrics.get("algorithm", "?")
    budget  = metrics.get("budget",    "?")
    seeds   = metrics.get("seeds",     "?")

    def fmt_int(v):   return f"{v:,}"     if isinstance(v, int)   else str(v)
    def fmt_pct(v):   return f"{v*100:.2f}%" if isinstance(v, float) else "—"
    def fmt_val(v):
        if v is None:             return "—"
        if isinstance(v, float):  return f"{v*100:.4f}%"
        if isinstance(v, int):    return f"{v:,}"
        return str(v)

    sep = "─" * 50
    print()
    print(sep)
    print(f"  TGA Evaluation Report — {algo}")
    print(sep)
    print(f"  Budget        : {fmt_int(budget)}")
    print(f"  Seeds         : {fmt_int(seeds)}")
    print(sep)
    print(f"  Probed        : {fmt_int(metrics.get('probed', 0))}")
    print(f"  Hits          : {fmt_int(metrics.get('hits', 0))}")
    print(f"  Hit rate      : {fmt_pct(metrics.get('hit_rate', 0.0))}")
    print(sep)
    print(f"  New hits      : {fmt_int(metrics.get('new_hits', 0))}")
    print(f"  New hit rate  : {fmt_pct(metrics.get('new_hit_rate', 0.0))}")
    print(sep)
    print(f"  /64 found     : {fmt_int(metrics.get('slash64_found', 0))}")
    print(f"  New /64s      : {fmt_int(metrics.get('new_slash64', 0))}")
    print(f"  New /64 rate  : {fmt_pct(metrics.get('new_slash64_rate', 0.0))}")
    if metrics.get("coverage") is not None:
        print(sep)
        print(f"  Coverage      : {fmt_pct(metrics.get('coverage', 0.0))}")
    print(sep)
    print()


def save_report(metrics: Dict[str, Any], path: str) -> None:
    """Persist metrics as JSON."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
