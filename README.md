# Target Generation Algorithms (TGA)

A unified Python + Rust implementation of IPv6 Target Generation Algorithms, with an integrated evaluation pipeline.

The project reproduces and extends algorithms studied in:

> *"Target Acquired? Evaluating Target Generation Algorithms for IPv6"* (TMA 2023)
> Reference: https://github.com/tumi8/tma-23-target-generation

**Goal:** re-implement all algorithms in a single, consistent codebase; pair them with a real end-to-end evaluation pipeline (address generation → ICMPv6 probing → metric reporting) so that results are directly comparable and reproducible.

---

## Algorithms

| Algorithm  | Original Language | Status         | Description |
|------------|------------------|----------------|-------------|
| 6tree      | C++              | **Complete**   | DHC tree — tree-based address generation |
| 6Forest    | Python           | Stub           | Space-partitioning forest-based TGA |
| 6GAN       | Python           | Stub           | GAN-based address generation |
| 6GCVAE     | Python           | Stub           | Graph convolutional variational autoencoder |
| 6Graph     | Python           | Stub           | Graph pattern mining TGA |
| 6Scan      | C/C++            | Stub           | Systematic IPv6 scanning |
| 6VecLM     | Python           | Stub           | Vector language model |
| DET        | Python           | Stub           | Dynamic entropy-based targeting |
| entropy-ip | Python/Bash      | Stub           | Entropy/Bayesian address generation |

---

## Project Structure

```
Target Generation Algorithms/
├── algorithms/
│   ├── six_tree/             # 6tree — complete
│   │   ├── translation.py    #   address format conversion (std/b1/b2/b3/b4/b5)
│   │   ├── tree.py           #   DHC space tree construction
│   │   ├── generation.py     #   density-ranked target generation
│   │   └── README.md
│   ├── six_forest/           # stub
│   ├── six_gan/              # stub
│   ├── six_gcvae/            # stub
│   ├── six_graph/            # stub
│   ├── six_scan/             # stub
│   ├── six_vec_lm/           # stub
│   ├── det/                  # stub
│   └── entropy_ip/           # stub
├── scanner/                  # Rust ICMPv6 prober
│   ├── Cargo.toml
│   └── src/main.rs           #   sender/receiver threads, 20 kpps, pnet
├── eval/                     # Python evaluation pipeline
│   ├── metrics.py            #   hit_rate, new_hit_rate, /64 discovery (TMA-23 + 6sense)
│   ├── pipeline.py           #   end-to-end orchestration (online & offline modes)
│   └── report.py             #   console + JSON output
├── data/cache/               # auto-downloaded aliased-prefixes.txt (gitignored)
├── main.py                   # unified algorithm entry point
├── requirements.txt
└── README.md
```

---

## Setup

### Python environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Rust scanner (Linux execution host only)

```bash
cd scanner
cargo build --release
# binary: scanner/target/release/scanner
# requires root / CAP_NET_RAW at runtime
```

---

## Usage

### Generate addresses only

```bash
python main.py --algorithm 6tree \
               --seeds data/seeds.txt \
               --output results/candidates.txt \
               --budget 1000000
```

| Argument      | Short | Description                                    |
|---------------|-------|------------------------------------------------|
| `--algorithm` | `-a`  | Algorithm name (see table above)               |
| `--seeds`     | `-s`  | Seed IPv6 address file (one per line)          |
| `--output`    | `-o`  | Output file for generated candidate addresses  |
| `--budget`    | `-b`  | Max addresses to generate (default: 1,000,000) |

### Full evaluation pipeline (online — requires Linux + public IPv6)

```bash
sudo python -m eval.pipeline \
    --algorithm 6tree \
    --seeds /data/seeds.txt \
    --budget 1000000 \
    --output-dir /data/results/6tree_run1 \
    --scanner ./scanner/target/release/scanner \
    --rate 20000 \
    --timeout 3
```

The pipeline:
1. Downloads and caches `aliased-prefixes.txt` from the TUM IPv6 hitlist service
2. Runs the chosen algorithm to generate candidate addresses
3. Filters out seed addresses and known aliased prefixes
4. Invokes the Rust scanner (ICMPv6 Echo, 20 kpps)
5. Computes and prints evaluation metrics

### Offline evaluation (no scanning needed)

```bash
python -m eval.pipeline \
    --algorithm 6tree \
    --seeds /data/seeds.txt \
    --budget 1000000 \
    --output-dir /data/results/6tree_offline \
    --offline \
    --hitlist /data/gasser_hitlist.txt
```

### Evaluation metrics

| Metric           | Description |
|------------------|-------------|
| `hit_rate`       | hits / probed |
| `new_hit_rate`   | (hits − seeds) / probed — truly new discoveries |
| `slash64_found`  | unique /64 prefixes among hits |
| `new_slash64`    | /64 prefixes in hits not present in seeds |
| `new_slash64_rate` | new_slash64 / slash64_found |
| `coverage`       | hits / ground-truth size (offline mode) |

Metrics are inspired by TMA-23 (hit rate, coverage) and 6sense (subnet-level discovery).

---

## Daily Progress

### 2026-02-24
- **6tree complete**: ported DHC space tree + density-ranked generation from C++ to Python (`algorithms/six_tree/`)
- **Rust scanner**: minimal ICMPv6 echo prober (`scanner/`), pnet-based sender/receiver threads, configurable rate (default 20 kpps), auto-stops after reply timeout
- **Evaluation pipeline**: `eval/` module with 6 metrics, aliased-prefix filtering with auto-download, online and offline modes
- Project pushed to GitHub; execution host is a Linux machine with 4090 GPU and public IPv6; full end-to-end run planned for next session

---

## Contributing

Each algorithm lives in its own subdirectory under `algorithms/` and must expose:

```python
def run(seeds: str, output: str, budget: int) -> None: ...
```

This is the interface called by `main.py` and `eval/pipeline.py`.
