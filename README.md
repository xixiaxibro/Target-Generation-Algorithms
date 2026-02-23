# Target Generation Algorithms (TGA) — Unified Python Implementation

This project provides a unified Python implementation of IPv6 Target Generation Algorithms (TGAs) originally studied in the TMA'23 paper:

> *"Target Acquired? Evaluating Target Generation Algorithms for IPv6"*
> Reference repository: https://github.com/tumi8/tma-23-target-generation

The goal of this project is to rewrite all algorithms in a single, consistent Python codebase for easier experimentation, comparison, and extension.

---

## Algorithms

| Algorithm   | Original Language | Status        | Description |
|-------------|------------------|---------------|-------------|
| 6Forest     | Python           | In progress   | Space-partitioning forest-based TGA |
| 6GAN        | Python           | In progress   | GAN-based address generation |
| 6GCVAE      | Python           | In progress   | Graph convolutional variational autoencoder |
| 6Graph      | Python           | In progress   | Graph pattern mining TGA |
| 6Scan       | C/C++            | To be ported  | Systematic IPv6 scanning |
| 6VecLM      | Python           | In progress   | Vector language model |
| 6tree       | C++              | To be ported  | Tree-based address generation |
| DET         | Python           | In progress   | Dynamic entropy-based targeting |
| entropy-ip  | Python/Bash      | In progress   | Entropy/Bayesian address generation |

---

## Project Structure

```
Target Generation Algorithms/
├── algorithms/
│   ├── six_forest/       # 6Forest
│   ├── six_gan/          # 6GAN
│   ├── six_gcvae/        # 6GCVAE
│   ├── six_graph/        # 6Graph
│   ├── six_scan/         # 6Scan (ported from C)
│   ├── six_vec_lm/       # 6VecLM
│   ├── six_tree/         # 6tree (ported from C++)
│   ├── det/              # DET
│   └── entropy_ip/       # entropy-ip
├── main.py               # Unified entry point
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py --algorithm <name> --seeds <seed_file> --output <output_file> [--budget N]
```

### Example

```bash
python main.py --algorithm 6tree --seeds data/seeds.txt --output results/targets.txt --budget 1000000
```

### Arguments

| Argument      | Short | Description                                      |
|---------------|-------|--------------------------------------------------|
| `--algorithm` | `-a`  | Algorithm name (see table above)                 |
| `--seeds`     | `-s`  | Path to seed IPv6 address file (one per line)    |
| `--output`    | `-o`  | Path to write generated target addresses         |
| `--budget`    | `-b`  | Max addresses to generate (default: 1,000,000)   |

---

## Contributing

Each algorithm lives in its own subdirectory under `algorithms/`. Each module must expose a `run(seeds, output, budget)` function that serves as the entry point called by `main.py`.
