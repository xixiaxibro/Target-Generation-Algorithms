"""
6GCVAE model: Gated Convolutional Variational Autoencoder for IPv6.

Architecture (ported from CuiTianyu961030/6GCVAE, gcnn_vae.py):

  Encoder:
    Embedding(16, emb_dim)
    → GatedConvBlock × 3  (increasing channels)
    → Flatten
    → Linear → z_mean (latent_dim)
    → Linear → z_log_var (latent_dim)

  Reparametrisation: z = z_mean + ε · exp(0.5 · z_log_var),  ε ~ N(0,I)

  Decoder:
    Linear(latent_dim → seq_len × n_filters)
    → Reshape (n_filters, seq_len)
    → GatedConvBlock × 3
    → Conv1d(vocab_size, 1) → (seq_len, vocab_size) logits

  Loss:
    recon  = CrossEntropy(logits, true_nibbles)
    kl     = −0.5 · Σ(1 + z_log_var − z_mean² − exp(z_log_var))
    total  = recon + beta · kl

API differences from original (Keras → PyTorch):
  - Session/graph execution replaced by eager PyTorch nn.Module.
  - No Keras Model.compile / model.fit; training loop handled in __init__.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defaults matching the original paper / repo
VOCAB_SIZE  = 16    # nibble values 0–15
SEQ_LEN     = 32    # nibble positions per address
LATENT_DIM  = 128
EMB_DIM     = 64
N_FILTERS   = 128


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class GatedConvBlock(nn.Module):
    """Gated convolutional layer with optional residual connection.

    Output = (W * x) ⊙ σ(V * x)

    A residual skip-connection is added when in_ch == out_ch, matching the
    "gated residual" formulation used in the original GCNN layer.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        # Double output channels: first half = linear, second half = gate
        self.conv = nn.Conv1d(in_ch, out_ch * 2, kernel_size, padding=padding)
        self.residual = in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_ch, seq_len)
        Returns:
            (batch, out_ch, seq_len)
        """
        h = self.conv(x)                                # (B, out_ch*2, L)
        h_lin, h_gate = h.chunk(2, dim=1)               # each (B, out_ch, L)
        out = h_lin * torch.sigmoid(h_gate)
        if self.residual:
            return out + x
        return out


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class GCVAEEncoder(nn.Module):
    """Encodes a 32-nibble sequence to (z_mean, z_log_var)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        emb_dim: int = EMB_DIM,
        n_filters: int = N_FILTERS,
        seq_len: int = SEQ_LEN,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.conv1 = GatedConvBlock(emb_dim,       n_filters,     kernel_size=3, padding=1)
        self.conv2 = GatedConvBlock(n_filters,     n_filters,     kernel_size=3, padding=1)
        self.conv3 = GatedConvBlock(n_filters,     n_filters * 2, kernel_size=3, padding=1)

        flat_dim = seq_len * (n_filters * 2)
        self.fc_mean    = nn.Linear(flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(flat_dim, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len) int64 nibble tokens.
        Returns:
            (z_mean, z_log_var): each (batch, latent_dim).
        """
        h = self.embedding(x).transpose(1, 2)   # (B, emb_dim, seq_len)
        h = self.conv1(h)                        # (B, n_filters,   seq_len)
        h = self.conv2(h)                        # (B, n_filters,   seq_len)
        h = self.conv3(h)                        # (B, n_filters*2, seq_len)
        h = h.flatten(1)                         # (B, seq_len * n_filters * 2)
        return self.fc_mean(h), self.fc_log_var(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class GCVAEDecoder(nn.Module):
    """Decodes a latent vector z to (batch, seq_len, vocab_size) logits."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        emb_dim: int = EMB_DIM,
        n_filters: int = N_FILTERS,
        seq_len: int = SEQ_LEN,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.n_filters = n_filters
        self.vocab_size = vocab_size

        self.fc = nn.Linear(latent_dim, seq_len * n_filters)

        self.conv1 = GatedConvBlock(n_filters,        n_filters,        kernel_size=3, padding=1)
        self.conv2 = GatedConvBlock(n_filters,        n_filters,        kernel_size=3, padding=1)
        self.conv3 = GatedConvBlock(n_filters,        vocab_size * 2,   kernel_size=3, padding=1)

        # Final pointwise projection to vocab logits
        self.proj = nn.Conv1d(vocab_size, vocab_size, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim) latent vector.
        Returns:
            (batch, seq_len, vocab_size) logits.
        """
        h = self.fc(z).view(-1, self.n_filters, self.seq_len)  # (B, n_filters, L)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)                                       # (B, vocab*2, L)
        h_lin, h_gate = h.chunk(2, dim=1)                       # each (B, vocab, L)
        h = h_lin * torch.sigmoid(h_gate)                       # gated (B, vocab, L)
        return self.proj(h).transpose(1, 2)                     # (B, L, vocab)


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------

class GCVAE(nn.Module):
    """Gated Convolutional Variational Autoencoder for IPv6 nibble sequences."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        emb_dim: int = EMB_DIM,
        n_filters: int = N_FILTERS,
        seq_len: int = SEQ_LEN,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = GCVAEEncoder(vocab_size, emb_dim, n_filters, seq_len, latent_dim)
        self.decoder = GCVAEDecoder(vocab_size, emb_dim, n_filters, seq_len, latent_dim)

    def reparameterise(
        self, mean: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterisation trick: z = mean + ε · std,  ε ~ N(0, I)."""
        std = torch.exp(0.5 * log_var)
        return mean + torch.randn_like(std) * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:    (batch, seq_len, vocab_size)
            z_mean:    (batch, latent_dim)
            z_log_var: (batch, latent_dim)
        """
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterise(z_mean, z_log_var)
        logits = self.decoder(z)
        return logits, z_mean, z_log_var


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    z_mean: torch.Tensor,
    z_log_var: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ELBO loss = reconstruction + beta * KL divergence.

    Args:
        logits:    (batch, seq_len, vocab_size) — decoder output.
        targets:   (batch, seq_len)             — true nibble tokens (0–15).
        z_mean:    (batch, latent_dim)
        z_log_var: (batch, latent_dim)
        beta:      KL weighting (1.0 = standard VAE; < 1 = β-VAE relaxation).

    Returns:
        (total_loss, recon_loss, kl_loss): all scalar tensors.
    """
    batch = logits.size(0)

    # Reconstruction: cross-entropy over every nibble position
    recon = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="sum",
    ) / batch

    # KL divergence: D_KL(q(z|x) || p(z)) = −½ Σ(1 + log_var − mean² − exp(log_var))
    kl = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / batch

    return recon + beta * kl, recon, kl
