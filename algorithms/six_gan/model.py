"""
6GAN model: LSTM Generator + CNN Discriminator (PyTorch port).

Ported from CuiTianyu961030/6GAN (TensorFlow original):
  - generator.py   → LSTMGenerator
  - discriminator.py → CNNDiscriminator

API differences:
  - TF session/variable graph → PyTorch nn.Module + autograd.
  - REINFORCE reward injection handled externally (see training.py).
  - Xavier initialisation applied uniformly via init_weights().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import BOS_TOKEN, EOS_TOKEN, VOCAB_SIZE, SEQ_LEN


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class LSTMGenerator(nn.Module):
    """Autoregressive LSTM sequence generator.

    Input  → (batch, seq_len) BOS-prepended token IDs.
    Output → (batch, seq_len, vocab_size) logits.

    During inference, tokens are drawn one at a time using .sample() which
    masks BOS/EOS to ensure only valid nibble tokens (0–15) are emitted.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        emb_dim: int = 200,
        hidden_dim: int = 200,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len    = seq_len

        self.embedding    = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm         = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x:      (batch, seq_len) token IDs.
            hidden: Optional (h_0, c_0) LSTM state.

        Returns:
            (logits, hidden): logits (batch, seq_len, vocab_size).
        """
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        return self.output_layer(out), hidden

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n: int,
        device: torch.device,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample *n* sequences of length *seq_len* (nibble tokens only).

        Returns:
            (n, seq_len) long tensor with values in [0, 15].
        """
        self.eval()
        inp    = torch.full((n, 1), BOS_TOKEN, dtype=torch.long, device=device)
        hidden = None
        tokens: list[torch.Tensor] = []

        for _ in range(self.seq_len):
            logits, hidden = self.forward(inp, hidden)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (n, vocab)
            # Mask special tokens — only emit nibbles 0-15
            logits[:, BOS_TOKEN:] = -1e9
            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, 1)              # (n, 1)
            tokens.append(next_tok)
            inp = next_tok

        return torch.cat(tokens, dim=1)  # (n, seq_len)

    @torch.no_grad()
    def rollout_complete(
        self,
        partial: torch.Tensor,
        rollout_num: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Complete a partial sequence *rollout_num* times.

        Used by the Monte Carlo reward estimator in training.py.

        Args:
            partial:     (batch, t) prefix tokens (t < seq_len).
            rollout_num: Number of independent completions per prefix.
            device:      Torch device.

        Returns:
            (batch * rollout_num, seq_len) completed sequences.
        """
        self.eval()
        batch, t = partial.size()
        remain   = self.seq_len - t

        # Expand prefix to batch * rollout_num
        expanded = partial.unsqueeze(1).expand(-1, rollout_num, -1)   # (B, R, t)
        expanded = expanded.reshape(batch * rollout_num, t)             # (B*R, t)

        if remain <= 0:
            return expanded[:, :self.seq_len]

        inp    = expanded
        hidden = None
        rest: list[torch.Tensor] = []

        for _ in range(remain):
            logits, hidden = self.forward(inp, hidden)
            logits = logits[:, -1, :]
            logits[:, BOS_TOKEN:] = -1e9
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)   # (B*R, 1)
            rest.append(next_tok)
            inp = next_tok

        return torch.cat([expanded] + rest, dim=1)[:, :self.seq_len]   # (B*R, seq_len)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class _Highway(nn.Module):
    """Single highway layer: T·relu(W·x) + (1−T)·x."""

    def __init__(self, size: int):
        super().__init__()
        self.transform = nn.Linear(size, size)
        self.gate      = nn.Linear(size, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(self.gate(x))
        return t * F.relu(self.transform(x)) + (1.0 - t) * x


class CNNDiscriminator(nn.Module):
    """CNN discriminator (real vs. generated nibble sequences).

    Architecture:
      Embedding → multiple parallel Conv1d (variable filter widths)
      → global max-pool → concat → highway × num_highway → Linear(2)

    Filter sizes and counts match the original 6GAN discriminator.
    """

    _FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    _NUM_FILTERS  = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160]

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        emb_dim: int = 64,
        seq_len: int = SEQ_LEN,
        num_highway: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        valid = [
            (k, n)
            for k, n in zip(self._FILTER_SIZES, self._NUM_FILTERS)
            if k <= seq_len
        ]
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, n_filt, k_size) for k_size, n_filt in valid
        ])
        total_filters = sum(n for _, n in valid)

        self.highway = nn.Sequential(*[_Highway(total_filters) for _ in range(num_highway)])
        self.dropout = nn.Dropout(dropout)
        self.output  = nn.Linear(total_filters, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token IDs.
        Returns:
            (batch, 2) logits [fake_logit, real_logit].
        """
        emb = self.embedding(x).transpose(1, 2)   # (B, emb_dim, seq_len)

        pooled = [
            F.relu(conv(emb)).max(dim=2).values    # (B, n_filters)
            for conv in self.convs
        ]
        h = torch.cat(pooled, dim=1)               # (B, total_filters)
        h = self.highway(h)
        h = self.dropout(h)
        return self.output(h)                       # (B, 2)


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module) -> None:
    """Apply Xavier uniform initialisation to all Linear and Conv1d layers."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
