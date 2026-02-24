"""
6GCVAE generation: sample from the VAE latent space to produce IPv6 targets.

Ported from CuiTianyu961030/6GCVAE (generation.py).

Generation procedure
--------------------
1. Sample z ~ N(0, I)  (standard normal, shape: batch × latent_dim)
2. Pass z through the trained decoder → (batch, seq_len, vocab_size) logits
3. Argmax over vocab → (batch, seq_len) nibble indices
4. Convert nibble indices to b4 hex strings → standard IPv6 notation
5. Deduplicate; repeat until *budget* unique addresses are collected.
"""

from __future__ import annotations

import torch

from algorithms.six_tree.translation import b4_to_std
from .model import GCVAE

_HEX = "0123456789abcdef"


def generate_addresses(
    model: GCVAE,
    budget: int,
    batch_size: int = 1024,
) -> list[str]:
    """Generate up to *budget* unique IPv6 addresses via VAE latent sampling.

    Args:
        model:      Trained GCVAE (eval mode expected; decoder is used directly).
        budget:     Target number of unique addresses to return.
        batch_size: Number of latent samples decoded per iteration.

    Returns:
        List of at most *budget* unique standard IPv6 address strings.
    """
    device = next(model.parameters()).device
    model.eval()

    results: set[str] = set()

    with torch.no_grad():
        while len(results) < budget:
            z      = torch.randn(batch_size, model.latent_dim, device=device)
            logits = model.decoder(z)                        # (B, seq_len, vocab_size)
            nibbles = logits.argmax(dim=-1).cpu().numpy()    # (B, seq_len)

            for row in nibbles:
                # Clamp to valid nibble range defensively (decoder output spans vocab)
                b4  = "".join(_HEX[int(n) % 16] for n in row)
                std = b4_to_std(b4)
                results.add(std)
                if len(results) >= budget:
                    break

    return list(results)[:budget]
