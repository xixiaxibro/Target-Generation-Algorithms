"""
6GCVAE: Gated Convolutional Variational Autoencoder for IPv6 target generation.

Paper: "Optimized IPv6 Target Generation via Network-wide IPv6 Address Relationship
        Mining Using GCVAE"  Li et al., PAKDD 2020.
Reference: https://github.com/tumi8/tma-23-target-generation/tree/main/algorithms/6GCVAE

Algorithm
---------
1. **Load** — seeds normalised to b4 nibble matrix (n, 32), values 0–15.

2. **Encode** — a Gated Convolutional encoder maps each 32-nibble sequence to
   (z_mean, z_log_var) in a latent space of dimension *latent_dim* = 128.

3. **Train** — minimise ELBO = reconstruction loss + β · KL divergence.
   Reconstruction: cross-entropy at every nibble position.
   KL: D_KL( q(z|x) || N(0,I) ).

4. **Generate** — sample z ~ N(0, I), decode through the Gated Convolutional
   decoder, argmax over vocab → 32 nibble indices → standard IPv6 strings.

5. **Cache** — trained model state-dict is saved to ``data/cache/6gcvae/``
   together with a SHA-256 fingerprint of the seed file.  Subsequent calls
   with identical seeds skip training.

Usage
-----
    from algorithms.six_gcvae import run
    run(seeds="data/seeds.txt", output="out.txt", budget=1_000_000)
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from algorithms.six_tree.translation import normalize_to_b4
from .generation import generate_addresses
from .model import GCVAE, vae_loss


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

CACHE_DIR   = Path("data/cache/6gcvae")
_MODEL_PATH = CACHE_DIR / "gcvae.pt"
_HASH_PATH  = CACHE_DIR / "seeds.sha256"

# ---------------------------------------------------------------------------
# Hyperparameters (matching original PAKDD 2020 paper / repo defaults)
# ---------------------------------------------------------------------------

_VOCAB_SIZE  = 16
_SEQ_LEN     = 32
_LATENT_DIM  = 128
_EMB_DIM     = 64
_N_FILTERS   = 128
_EPOCHS      = 50
_BATCH_SIZE  = 128
_LR          = 1e-3
_BETA        = 1.0   # KL weight (standard VAE)
_GEN_BATCH   = 1024  # latent samples per generation iteration


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(seeds: str, output: str, budget: int) -> None:
    """Generate IPv6 target addresses using 6GCVAE.

    Args:
        seeds:  Path to seed IPv6 address file (one per line).
        output: Path to write generated target addresses.
        budget: Maximum number of addresses to generate.
    """
    print(f"[6GCVAE] Loading seeds from: {seeds}")
    b4_seeds = _load_seeds(seeds)
    if not b4_seeds:
        print("[6GCVAE] ERROR: no valid seed addresses — aborting.")
        return
    print(f"[6GCVAE] {len(b4_seeds):,} unique seeds loaded.")

    arrs = np.array(
        [[int(c, 16) for c in addr] for addr in b4_seeds], dtype=np.uint8
    )

    # ── Cache check ──────────────────────────────────────────────────────────
    seed_hash = _sha256_file(seeds)
    model     = _try_load_cache(seed_hash)

    if model is None:
        # ── Train ────────────────────────────────────────────────────────────
        print("[6GCVAE] Training GCVAE …")
        model = _train(arrs)
        _save_cache(model, seed_hash)
        print(f"[6GCVAE] Model cached to: {CACHE_DIR}")
    else:
        print("[6GCVAE] Cache hit — skipping training.")

    # ── Generate ─────────────────────────────────────────────────────────────
    print(f"[6GCVAE] Generating up to {budget:,} target addresses …")
    targets = generate_addresses(model, budget, batch_size=_GEN_BATCH)
    print(f"[6GCVAE] {len(targets):,} addresses generated.")

    with open(output, "w") as fh:
        for addr in targets:
            fh.write(addr + "\n")
    print(f"[6GCVAE] Targets written to: {output}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(arrs: np.ndarray) -> GCVAE:
    """Train the GCVAE on the nibble matrix and return the trained model."""
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[6GCVAE] Using device: {device}")

    model = GCVAE(
        vocab_size=_VOCAB_SIZE,
        emb_dim=_EMB_DIM,
        n_filters=_N_FILTERS,
        seq_len=_SEQ_LEN,
        latent_dim=_LATENT_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=_LR)
    dataset   = TensorDataset(torch.tensor(arrs, dtype=torch.long))
    loader    = DataLoader(dataset, batch_size=_BATCH_SIZE, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(_EPOCHS):
        total_loss = total_recon = total_kl = 0.0
        n_batches  = 0

        for (x,) in loader:
            x = x.to(device)
            logits, z_mean, z_log_var = model(x)
            loss, recon, kl = vae_loss(logits, x, z_mean, z_log_var, beta=_BETA)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss  += loss.item()
            total_recon += recon.item()
            total_kl    += kl.item()
            n_batches   += 1

        n = max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[6GCVAE]   Epoch {epoch + 1:3d}/{_EPOCHS}"
                f" — loss: {total_loss/n:.4f}"
                f"  recon: {total_recon/n:.4f}"
                f"  kl: {total_kl/n:.4f}"
            )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_load_cache(seed_hash: str) -> GCVAE | None:
    if not (_MODEL_PATH.exists() and _HASH_PATH.exists()):
        return None
    if _HASH_PATH.read_text().strip() != seed_hash:
        print("[6GCVAE] Seed file changed — invalidating cache.")
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = GCVAE(
            vocab_size=_VOCAB_SIZE, emb_dim=_EMB_DIM, n_filters=_N_FILTERS,
            seq_len=_SEQ_LEN, latent_dim=_LATENT_DIM,
        ).to(device)
        state = torch.load(str(_MODEL_PATH), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        print(f"[6GCVAE] Loaded cached model from: {CACHE_DIR}")
        return model
    except Exception as exc:
        print(f"[6GCVAE] Cache load failed ({exc}) — retraining.")
        return None


def _save_cache(model: GCVAE, seed_hash: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(_MODEL_PATH))
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
