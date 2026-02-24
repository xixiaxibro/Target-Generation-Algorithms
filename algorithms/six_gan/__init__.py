"""
6GAN: Generative Adversarial Network for IPv6 target generation.

Paper: "6GAN: IPv6 Multi-Pattern Target Generation via Generative Adversarial
        Nets with Random Walk"  Cui et al., IEEE INFOCOM 2021.
Reference: https://github.com/tumi8/tma-23-target-generation/tree/main/algorithms/6GAN

Algorithm
---------
1. **Load** — seeds normalised to b4 nibble matrix (n, 32), values 0–15.

2. **Pre-train Generator** (MLE, 50 epochs) — teacher-forcing cross-entropy
   trains the LSTM to model P(nibble_t | BOS, nibble_0 … nibble_{t-1}).
   This stabilises adversarial training by giving the generator a good
   starting point.

3. **Pre-train Discriminator** (10 iterations) — the CNN discriminator learns
   to distinguish real seed addresses from generator samples.

4. **Adversarial Training** (SeqGAN, 200 mini-batches):
   • Generator: REINFORCE with per-step Monte Carlo rewards.
     At step t, *rollout_num* (=16) completions are drawn from the generator
     and scored by the discriminator's P(real); the mean is reward R_t.
     Loss = −Σ_t R_t · log π_θ(a_t | s_{<t})
   • Discriminator: MLE on fresh real + generated batches (3 updates/step).

5. **Generate** — sample sequences from the trained generator and convert to
   standard IPv6 notation.

6. **Cache** — the trained generator state-dict is saved to
   ``data/cache/6gan/`` with a SHA-256 seed fingerprint.

Usage
-----
    from algorithms.six_gan import run
    run(seeds="data/seeds.txt", output="out.txt", budget=1_000_000)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from algorithms.six_tree.translation import b4_to_std, normalize_to_b4
from .data import HEX_CHARS, SEQ_LEN, make_gen_dataloader
from .model import CNNDiscriminator, LSTMGenerator, init_weights
from .training import adversarial_train, discriminator_pretrain, mle_pretrain


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

CACHE_DIR   = Path("data/cache/6gan")
_GEN_PATH   = CACHE_DIR / "generator.pt"
_HASH_PATH  = CACHE_DIR / "seeds.sha256"

# ---------------------------------------------------------------------------
# Hyperparameters (matching the original 6GAN paper / repo)
# ---------------------------------------------------------------------------

_EMB_DIM        = 200
_HIDDEN_DIM     = 200
_MLE_EPOCHS     = 50
_DIS_PRETRAIN   = 10
_ADV_BATCHES    = 200
_BATCH_SIZE     = 64
_ROLLOUT_NUM    = 16
_G_LR           = 1e-4
_D_LR           = 1e-4
_GEN_TEMPERATURE = 1.0   # sampling temperature at generation time


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(seeds: str, output: str, budget: int) -> None:
    """Generate IPv6 target addresses using 6GAN.

    Args:
        seeds:  Path to seed IPv6 address file (one per line).
        output: Path to write generated target addresses.
        budget: Maximum number of addresses to generate.
    """
    print(f"[6GAN] Loading seeds from: {seeds}")
    b4_seeds = _load_seeds(seeds)
    if not b4_seeds:
        print("[6GAN] ERROR: no valid seed addresses — aborting.")
        return
    print(f"[6GAN] {len(b4_seeds):,} unique seeds loaded.")

    arrs = np.array(
        [[int(c, 16) for c in addr] for addr in b4_seeds], dtype=np.uint8
    )

    # ── Cache check ──────────────────────────────────────────────────────────
    seed_hash = _sha256_file(seeds)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = _try_load_cache(seed_hash, device)

    if generator is None:
        # ── Train ────────────────────────────────────────────────────────────
        print(f"[6GAN] Using device: {device}")
        generator, _ = _train(arrs, device)
        _save_cache(generator, seed_hash)
        print(f"[6GAN] Generator cached to: {CACHE_DIR}")
    else:
        print("[6GAN] Cache hit — skipping training.")

    # ── Generate ─────────────────────────────────────────────────────────────
    print(f"[6GAN] Generating up to {budget:,} target addresses …")
    targets = _generate(generator, budget, device)
    print(f"[6GAN] {len(targets):,} addresses generated.")

    with open(output, "w") as fh:
        for addr in targets:
            fh.write(addr + "\n")
    print(f"[6GAN] Targets written to: {output}")


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def _train(
    arrs: np.ndarray,
    device: torch.device,
) -> tuple[LSTMGenerator, CNNDiscriminator]:
    """Run the full three-phase 6GAN training schedule."""
    from .data import VOCAB_SIZE

    generator     = LSTMGenerator(VOCAB_SIZE, _EMB_DIM, _HIDDEN_DIM, SEQ_LEN).to(device)
    discriminator = CNNDiscriminator(VOCAB_SIZE, seq_len=SEQ_LEN).to(device)
    init_weights(generator)
    init_weights(discriminator)

    gen_loader = make_gen_dataloader(arrs, batch_size=_BATCH_SIZE)

    # Phase 1: Generator MLE pre-training
    print(f"[6GAN] Phase 1 — MLE pre-training ({_MLE_EPOCHS} epochs) …")
    mle_pretrain(generator, gen_loader, device, epochs=_MLE_EPOCHS)

    # Phase 2: Discriminator pre-training
    print(f"[6GAN] Phase 2 — Discriminator pre-training ({_DIS_PRETRAIN} iters) …")
    discriminator_pretrain(
        discriminator, generator, arrs, device,
        iterations=_DIS_PRETRAIN, batch_size=_BATCH_SIZE,
    )

    # Phase 3: Adversarial (SeqGAN)
    print(f"[6GAN] Phase 3 — Adversarial training ({_ADV_BATCHES} batches) …")
    adversarial_train(
        generator, discriminator, arrs, device,
        total_batches=_ADV_BATCHES,
        batch_size=_BATCH_SIZE,
        rollout_num=_ROLLOUT_NUM,
        g_lr=_G_LR,
        d_lr=_D_LR,
    )

    generator.eval()
    discriminator.eval()
    return generator, discriminator


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(
    generator: LSTMGenerator,
    budget: int,
    device: torch.device,
    batch_size: int = 2048,
) -> list[str]:
    """Sample sequences from the generator and convert to IPv6 strings."""
    results: set[str] = set()

    while len(results) < budget:
        n    = min(batch_size, (budget - len(results)) * 2)
        seqs = generator.sample(n, device, temperature=_GEN_TEMPERATURE)
        for row in seqs.cpu().numpy():
            b4  = "".join(HEX_CHARS[int(v) % 16] for v in row)
            std = b4_to_std(b4)
            results.add(std)
            if len(results) >= budget:
                break

    return list(results)[:budget]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_load_cache(seed_hash: str, device: torch.device) -> LSTMGenerator | None:
    if not (_GEN_PATH.exists() and _HASH_PATH.exists()):
        return None
    if _HASH_PATH.read_text().strip() != seed_hash:
        print("[6GAN] Seed file changed — invalidating cache.")
        return None
    try:
        from .data import VOCAB_SIZE
        gen   = LSTMGenerator(VOCAB_SIZE, _EMB_DIM, _HIDDEN_DIM, SEQ_LEN).to(device)
        state = torch.load(str(_GEN_PATH), map_location=device, weights_only=True)
        gen.load_state_dict(state)
        gen.eval()
        print(f"[6GAN] Loaded cached generator from: {CACHE_DIR}")
        return gen
    except Exception as exc:
        print(f"[6GAN] Cache load failed ({exc}) — retraining.")
        return None


def _save_cache(generator: LSTMGenerator, seed_hash: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), str(_GEN_PATH))
    _HASH_PATH.write_text(seed_hash)


# ---------------------------------------------------------------------------
# Seed loader
# ---------------------------------------------------------------------------

def _load_seeds(path: str) -> list[str]:
    b4_set: set[str] = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            b4 = normalize_to_b4(line)
            if b4 is not None:
                b4_set.add(b4)
    return sorted(b4_set)
