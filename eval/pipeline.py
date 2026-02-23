"""
End-to-end TGA evaluation pipeline.

Two modes
---------
Online (default):
    seeds → algorithm → filter → Rust scanner → hits → metrics

Offline (--offline):
    seeds → algorithm → filter → intersect with hitlist → metrics
    No network access needed; useful for quick iteration.

Filtering steps (both modes):
    1. Remove seed addresses from candidates  (only novel addresses count)
    2. Remove known aliased prefixes          (avoids inflated hit rates)

Aliased-prefixes list is downloaded automatically from the TUM IPv6 hitlist
service and cached locally.  Pass --no-refresh to skip re-downloading.

Usage examples
--------------
# Online — full pipeline (build scanner first: cd scanner && cargo build --release)
python -m eval.pipeline \\
    --algorithm 6tree \\
    --seeds /data/seeds.txt \\
    --budget 1000000 \\
    --output-dir /data/results/6tree \\
    --scanner ./scanner/target/release/scanner

# Offline — compare against Gasser hitlist without scanning
python -m eval.pipeline \\
    --algorithm 6tree \\
    --seeds /data/seeds.txt \\
    --budget 1000000 \\
    --output-dir /data/results/6tree \\
    --offline \\
    --hitlist /data/gasser_hitlist.txt
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import lzma
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

from .metrics import (
    build_aliased_index,
    compute_metrics,
    filter_aliased,
    slash64_set,
)
from .report import print_report, save_report

# ─── Constants ────────────────────────────────────────────────────────────────

ALIASED_URL = (
    "https://alcatraz.net.in.tum.de/ipv6-hitlist-service/open/"
    "aliased-prefixes.txt.xz"
)
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")


# ─── Aliased-prefixes management ──────────────────────────────────────────────

def _download_aliased_prefixes(cache_dir: str, url: str = ALIASED_URL) -> str:
    """Download and decompress aliased-prefixes.txt, return local path."""
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, "aliased-prefixes.txt")

    print(f"[pipeline] Downloading aliased prefixes from {url} …")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            compressed = resp.read()
        data = lzma.decompress(compressed)
        with open(dest, "wb") as f:
            f.write(data)
        print(f"[pipeline] Saved to {dest} ({len(data) // 1024} KB)")
    except Exception as e:
        print(f"[pipeline] WARNING: could not download aliased prefixes: {e}")
        if os.path.exists(dest):
            print(f"[pipeline] Using cached copy: {dest}")
        else:
            print("[pipeline] No cached copy available — skipping aliased filter.")
            return ""

    return dest


def load_aliased_index(path: str) -> dict:
    """Load aliased-prefixes.txt and return the pre-built lookup index."""
    if not path or not os.path.exists(path):
        return {}
    networks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                networks.append(ipaddress.IPv6Network(line, strict=False))
            except ValueError:
                pass
    return build_aliased_index(networks)


# ─── Seed / candidate file helpers ───────────────────────────────────────────

def load_addresses(path: str) -> list[str]:
    """Read one IPv6 address per line; skip blanks and comments."""
    result = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                result.append(line)
    return result


def filter_candidates(
    candidates: list[str],
    seeds_set:  set[str],
    aliased_index: dict,
) -> list[str]:
    """Remove seeds and aliased addresses from the candidate list."""
    # Step 1: remove seeds (only novel addresses are interesting).
    novel = [a for a in candidates if a not in seeds_set]
    removed_seeds = len(candidates) - len(novel)

    # Step 2: remove aliased prefixes.
    if aliased_index:
        clean = filter_aliased(novel, aliased_index)
        removed_aliased = len(novel) - len(clean)
    else:
        clean = novel
        removed_aliased = 0

    print(
        f"[pipeline] Filter: removed {removed_seeds:,} seeds, "
        f"{removed_aliased:,} aliased → {len(clean):,} candidates remain."
    )
    return clean


# ─── Algorithm runner ─────────────────────────────────────────────────────────

def run_algorithm(
    algorithm: str,
    seeds_path: str,
    output_path: str,
    budget: int,
) -> None:
    """Invoke the TGA via main.py and wait for completion."""
    cmd = [
        sys.executable, "main.py",
        "--algorithm", algorithm,
        "--seeds",     seeds_path,
        "--output",    output_path,
        "--budget",    str(budget),
    ]
    print(f"[pipeline] Running algorithm: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[pipeline] Algorithm done in {time.time() - t0:.1f}s.")


# ─── Scanner runner ───────────────────────────────────────────────────────────

def run_scanner(
    binary:      str,
    input_path:  str,
    output_path: str,
    rate:        int,
    timeout:     int,
) -> None:
    """Run the Rust ICMPv6 scanner as a subprocess."""
    cmd = [
        binary,
        "--input",   input_path,
        "--output",  output_path,
        "--rate",    str(rate),
        "--timeout", str(timeout),
    ]
    print(f"[pipeline] Running scanner: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[pipeline] Scanner done in {time.time() - t0:.1f}s.")


# ─── Offline evaluation ───────────────────────────────────────────────────────

def offline_evaluate(
    candidates: list[str],
    hitlist_path: str,
) -> tuple[list[str], list[str]]:
    """Return (probed, hits) by intersecting candidates with a hitlist."""
    print(f"[pipeline] Loading hitlist for offline eval: {hitlist_path}")
    hitlist = set(load_addresses(hitlist_path))
    hits = [a for a in candidates if a in hitlist]
    print(f"[pipeline] Offline: {len(hits):,} hits out of {len(candidates):,} candidates.")
    return candidates, hits


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    algorithm:    str,
    seeds_path:   str,
    budget:       int,
    output_dir:   str,
    scanner:      Optional[str] = None,
    rate:         int           = 20_000,
    timeout:      int           = 3,
    offline:      bool          = False,
    hitlist_path: Optional[str] = None,
    no_refresh:   bool          = False,
    cache_dir:    str           = DEFAULT_CACHE_DIR,
) -> dict:
    """Run the full TGA evaluation pipeline and return the metrics dict."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Paths ────────────────────────────────────────────────────────────────
    candidates_raw_path  = os.path.join(output_dir, "candidates_raw.txt")
    candidates_filt_path = os.path.join(output_dir, "candidates_filtered.txt")
    hits_path            = os.path.join(output_dir, "hits.txt")
    metrics_path         = os.path.join(output_dir, "metrics.json")

    # ── Aliased-prefixes ─────────────────────────────────────────────────────
    if no_refresh:
        aliased_path = os.path.join(cache_dir, "aliased-prefixes.txt")
    else:
        aliased_path = _download_aliased_prefixes(cache_dir)
    aliased_index = load_aliased_index(aliased_path)

    # ── Load seeds ───────────────────────────────────────────────────────────
    print(f"[pipeline] Loading seeds: {seeds_path}")
    seeds     = load_addresses(seeds_path)
    seeds_set = set(seeds)
    print(f"[pipeline] {len(seeds_set):,} unique seed addresses.")

    # ── Run algorithm ─────────────────────────────────────────────────────────
    run_algorithm(algorithm, seeds_path, candidates_raw_path, budget)
    candidates_raw = load_addresses(candidates_raw_path)
    print(f"[pipeline] Algorithm generated {len(candidates_raw):,} candidates.")

    # ── Filter candidates ────────────────────────────────────────────────────
    candidates = filter_candidates(candidates_raw, seeds_set, aliased_index)

    with open(candidates_filt_path, "w") as f:
        for addr in candidates:
            f.write(addr + "\n")

    # ── Probe / evaluate ─────────────────────────────────────────────────────
    if offline:
        if not hitlist_path:
            raise ValueError("--hitlist required in offline mode.")
        probed, hits = offline_evaluate(candidates, hitlist_path)
    else:
        if not scanner:
            raise ValueError("--scanner required in online mode.")
        run_scanner(scanner, candidates_filt_path, hits_path, rate, timeout)
        probed = candidates
        hits   = load_addresses(hits_path)

    # ── Compute metrics ───────────────────────────────────────────────────────
    ground_truth = set(load_addresses(hitlist_path)) if hitlist_path else None
    metrics = compute_metrics(hits, probed, seeds, ground_truth)
    metrics["algorithm"] = algorithm
    metrics["budget"]    = budget
    metrics["seeds"]     = len(seeds_set)

    # ── Report & save ─────────────────────────────────────────────────────────
    print_report(metrics)
    save_report(metrics, metrics_path)
    print(f"[pipeline] Metrics saved to {metrics_path}")

    return metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IPv6 TGA end-to-end evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--algorithm", "-a", required=True,
                   help="Algorithm name (e.g. 6tree)")
    p.add_argument("--seeds", "-s", required=True,
                   help="Seed IPv6 address file (one per line)")
    p.add_argument("--budget", "-b", type=int, default=1_000_000,
                   help="Max addresses to generate (default: 1,000,000)")
    p.add_argument("--output-dir", "-o", required=True,
                   help="Directory for all output files")

    # Online mode
    p.add_argument("--scanner", default=None,
                   help="Path to compiled Rust scanner binary (online mode)")
    p.add_argument("--rate", type=int, default=20_000,
                   help="Scanner packets per second (default: 20000)")
    p.add_argument("--timeout", type=int, default=3,
                   help="Scanner reply timeout in seconds (default: 3)")

    # Offline mode
    p.add_argument("--offline", action="store_true",
                   help="Offline mode: compare against --hitlist instead of scanning")
    p.add_argument("--hitlist", default=None,
                   help="Reference hitlist for offline eval or coverage computation")

    # Aliased prefixes
    p.add_argument("--no-refresh", action="store_true",
                   help="Skip downloading aliased-prefixes.txt; use cached copy")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                   help="Local cache directory for downloaded data")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        algorithm    = args.algorithm,
        seeds_path   = args.seeds,
        budget       = args.budget,
        output_dir   = args.output_dir,
        scanner      = args.scanner,
        rate         = args.rate,
        timeout      = args.timeout,
        offline      = args.offline,
        hitlist_path = args.hitlist,
        no_refresh   = args.no_refresh,
        cache_dir    = args.cache_dir,
    )
