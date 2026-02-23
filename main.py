"""
Target Generation Algorithms — unified Python entry point.

Usage:
    python main.py --algorithm <name> --seeds <seed_file> --output <output_file> [options]

Available algorithms:
    6forest     Space-partitioning forest-based TGA
    6gan        GAN-based IPv6 address generation
    6gcvae      Graph convolutional variational autoencoder
    6graph      Graph-pattern-mining TGA
    6scan       Systematic scanning (rewritten from C)
    6veclm      Vector language model for IPv6
    6tree       Tree-based address generation (rewritten from C++)
    det         Dynamic entropy-based targeting
    entropy-ip  Entropy/Bayesian address generation
"""

import argparse
import sys


ALGORITHMS = [
    "6forest",
    "6gan",
    "6gcvae",
    "6graph",
    "6scan",
    "6veclm",
    "6tree",
    "det",
    "entropy-ip",
]


def get_algorithm(name: str):
    """Import and return the run() function for the requested algorithm."""
    mapping = {
        "6forest":    "algorithms.six_forest",
        "6gan":       "algorithms.six_gan",
        "6gcvae":     "algorithms.six_gcvae",
        "6graph":     "algorithms.six_graph",
        "6scan":      "algorithms.six_scan",
        "6veclm":     "algorithms.six_vec_lm",
        "6tree":      "algorithms.six_tree",
        "det":        "algorithms.det",
        "entropy-ip": "algorithms.entropy_ip",
    }
    module_path = mapping.get(name)
    if module_path is None:
        print(f"Unknown algorithm '{name}'. Choose from: {', '.join(ALGORITHMS)}")
        sys.exit(1)
    import importlib
    module = importlib.import_module(module_path)
    return module.run


def parse_args():
    parser = argparse.ArgumentParser(
        description="IPv6 Target Generation Algorithms — unified Python interface"
    )
    parser.add_argument(
        "--algorithm", "-a",
        required=True,
        choices=ALGORITHMS,
        help="Algorithm to run",
    )
    parser.add_argument(
        "--seeds", "-s",
        required=True,
        help="Path to seed IPv6 address file (one address per line)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to write generated target addresses",
    )
    parser.add_argument(
        "--budget", "-b",
        type=int,
        default=1_000_000,
        help="Maximum number of addresses to generate (default: 1,000,000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run = get_algorithm(args.algorithm)
    print(f"[*] Running algorithm: {args.algorithm}")
    print(f"[*] Seeds file:        {args.seeds}")
    print(f"[*] Output file:       {args.output}")
    print(f"[*] Budget:            {args.budget:,}")
    run(seeds=args.seeds, output=args.output, budget=args.budget)
    print("[+] Done.")


if __name__ == "__main__":
    main()
