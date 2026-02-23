"""
IPv6 TGA evaluation metrics.

Metrics
-------
Standard (from TMA-23):
  hit_rate         hits / probed
  coverage         hits / |ground_truth|   (offline mode only)

6sense-inspired:
  new_hit_rate     (hits − seeds) / probed  — truly new discoveries
  slash64_found    unique /64 prefixes in hits
  new_slash64      /64 prefixes in hits NOT already seen in seeds
  new_slash64_rate new_slash64 / slash64_found

All functions are pure (no I/O).  Input collections may be any iterable of
canonical IPv6 strings (full colon-hex notation, lower-case preferred).
"""

from __future__ import annotations

import ipaddress
from typing import Collection, Dict, Iterable, Optional


# ─── Address helpers ──────────────────────────────────────────────────────────

def _addr(s: str) -> ipaddress.IPv6Address:
    return ipaddress.IPv6Address(s)


def addr_to_slash64(addr: str) -> str:
    """Return the /64 network prefix string for *addr*.

    Example:
        addr_to_slash64("2001:db8:1:2:3:4:5:6") → "2001:db8:1:2::"
    """
    net = ipaddress.IPv6Network(f"{addr}/64", strict=False)
    return str(net.network_address)


def slash64_set(addrs: Iterable[str]) -> set[str]:
    """Return the set of /64 prefixes covered by *addrs*."""
    result: set[str] = set()
    for a in addrs:
        try:
            result.add(addr_to_slash64(a))
        except ValueError:
            pass
    return result


# ─── Aliased-prefix filter ────────────────────────────────────────────────────

def build_aliased_index(
    aliased_networks: list[ipaddress.IPv6Network],
) -> dict[int, set[int]]:
    """Pre-process aliased prefixes into a length → set-of-network-ints index
    for O(K) per-address lookup (K = number of distinct prefix lengths).
    """
    index: dict[int, set[int]] = {}
    for net in aliased_networks:
        index.setdefault(net.prefixlen, set()).add(int(net.network_address))
    return index


def is_aliased(addr_int: int, index: dict[int, set[int]]) -> bool:
    """Return True if *addr_int* falls inside any aliased prefix."""
    for pfxlen, net_set in index.items():
        mask = ((1 << 128) - 1) ^ ((1 << (128 - pfxlen)) - 1)
        if addr_int & mask in net_set:
            return True
    return False


def filter_aliased(
    addrs: Iterable[str],
    index: dict[int, set[int]],
) -> list[str]:
    """Return addresses from *addrs* that are NOT in any aliased prefix."""
    result = []
    for a in addrs:
        try:
            if not is_aliased(int(_addr(a)), index):
                result.append(a)
        except ValueError:
            pass
    return result


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(
    hits:         Iterable[str],
    probed:       Iterable[str],
    seeds:        Iterable[str],
    ground_truth: Optional[Collection[str]] = None,
) -> Dict[str, object]:
    """Compute all TGA evaluation metrics.

    Parameters
    ----------
    hits:
        Addresses that responded to probes (online mode) or that appear in
        the ground-truth hitlist (offline mode).
    probed:
        All candidate addresses that were sent to the scanner (or checked
        against the hitlist).
    seeds:
        The original seed addresses used as algorithm input.
    ground_truth:
        (Offline mode) Full reference hitlist used to compute *coverage*.
        Leave as None for online mode.

    Returns
    -------
    dict with keys:
        probed          int   — total candidates evaluated
        hits            int   — responsive addresses
        hit_rate        float — hits / probed
        new_hits        int   — hits not present in seeds
        new_hit_rate    float — new_hits / probed
        slash64_found   int   — unique /64 prefixes among hits
        new_slash64     int   — /64 prefixes in hits but not in seeds
        new_slash64_rate float — new_slash64 / slash64_found  (0 if 0 found)
        coverage        float — hits / |ground_truth|  (None if offline not used)
    """
    hits_set   = set(hits)
    probed_set = set(probed)
    seeds_set  = set(seeds)

    n_probed = len(probed_set)
    n_hits   = len(hits_set)

    hit_rate = n_hits / n_probed if n_probed else 0.0

    new_hits     = hits_set - seeds_set
    n_new_hits   = len(new_hits)
    new_hit_rate = n_new_hits / n_probed if n_probed else 0.0

    s64_hits    = slash64_set(hits_set)
    s64_seeds   = slash64_set(seeds_set)
    n_s64_found = len(s64_hits)
    n_new_s64   = len(s64_hits - s64_seeds)
    new_s64_rate = n_new_s64 / n_s64_found if n_s64_found else 0.0

    coverage: Optional[float] = None
    if ground_truth is not None:
        n_gt = len(ground_truth)
        coverage = len(hits_set & set(ground_truth)) / n_gt if n_gt else 0.0

    return {
        "probed":          n_probed,
        "hits":            n_hits,
        "hit_rate":        hit_rate,
        "new_hits":        n_new_hits,
        "new_hit_rate":    new_hit_rate,
        "slash64_found":   n_s64_found,
        "new_slash64":     n_new_s64,
        "new_slash64_rate": new_s64_rate,
        "coverage":        coverage,
    }


# ─── Budget curve ─────────────────────────────────────────────────────────────

def budget_curve(
    hits_ordered: list[str],
    seeds_set:    set[str],
    budgets:      list[int],
) -> list[Dict[str, object]]:
    """Compute metrics at each budget checkpoint.

    *hits_ordered* must be the hit list **in the order they were discovered**
    (i.e., the order the scanner produced them, which approximates probe order).

    Returns a list of dicts, one per budget value, each containing
    ``budget``, ``hits``, ``hit_rate``, ``new_hits``, ``slash64_found``.
    """
    rows = []
    seen_hits: set[str]    = set()
    seen_s64:  set[str]    = set()
    hit_iter = iter(hits_ordered)
    budget_hits = 0  # hits consumed so far

    for budget in sorted(budgets):
        # Advance through hits up to this budget's position.
        # (We assume hits_ordered is already a subset of probed[0:budget].)
        while budget_hits < budget:
            try:
                h = next(hit_iter)
                seen_hits.add(h)
                seen_s64.add(addr_to_slash64(h))
                budget_hits += 1
            except StopIteration:
                break

        n_hits    = len(seen_hits)
        n_new     = len(seen_hits - seeds_set)
        hit_rate  = n_hits / budget if budget else 0.0
        rows.append({
            "budget":        budget,
            "hits":          n_hits,
            "hit_rate":      hit_rate,
            "new_hits":      n_new,
            "slash64_found": len(seen_s64),
        })

    return rows
