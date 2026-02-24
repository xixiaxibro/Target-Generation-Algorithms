"""
6GAN data: vocabulary, sequence encoding, DataLoader helpers.

Vocabulary (18 tokens)
----------------------
  0 – 15 : nibble hex values (0='0', 1='1', …, 10='a', …, 15='f')
  16     : BOS (begin-of-sequence / start token)
  17     : EOS (end-of-sequence)

Sequence formats
----------------
  discriminator input : (batch, SEQ_LEN=32)    — raw nibble tokens 0-15
  generator input     : (batch, SEQ_LEN=32)    — BOS-prepended, last nibble dropped
  generator target    : (batch, SEQ_LEN=32)    — original nibbles (teacher forcing)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HEX_CHARS  = "0123456789abcdef"
BOS_TOKEN  = 16
EOS_TOKEN  = 17
VOCAB_SIZE = 18
SEQ_LEN    = 32   # nibbles per address


# ---------------------------------------------------------------------------
# Sequence encoding
# ---------------------------------------------------------------------------

def arrs_to_seqs(arrs: np.ndarray) -> torch.Tensor:
    """Convert nibble matrix (n, 32) uint8 to (n, 32) long tensor."""
    return torch.tensor(arrs, dtype=torch.long)


def seqs_to_b4(seqs: torch.Tensor) -> list[str]:
    """Convert (n, 32) nibble tensor to list of 32-char b4 strings."""
    arr = seqs.cpu().numpy()
    return ["".join(HEX_CHARS[arr[i, j] % 16] for j in range(SEQ_LEN)) for i in range(len(arr))]


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def make_gen_dataloader(
    arrs: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader for generator MLE pre-training (teacher-forcing).

    Each sample pair:
        inp[i] = [BOS, nibble_0, …, nibble_30]   (length 32, BOS-prepended)
        tgt[i] = [nibble_0, nibble_1, …, nibble_31]  (length 32, shifted)
    """
    n = len(arrs)
    bos_col = np.full((n, 1), BOS_TOKEN, dtype=np.int64)
    inp = np.concatenate([bos_col, arrs[:, :-1].astype(np.int64)], axis=1)
    tgt = arrs.astype(np.int64)

    dataset = TensorDataset(
        torch.tensor(inp, dtype=torch.long),
        torch.tensor(tgt, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def make_dis_dataloader(
    real_arrs: np.ndarray,
    fake_arrs: np.ndarray,
    batch_size: int = 64,
) -> DataLoader:
    """DataLoader for discriminator training (real = 1, generated = 0)."""
    real_t = torch.tensor(real_arrs, dtype=torch.long)
    fake_t = torch.tensor(fake_arrs, dtype=torch.long)

    real_labels = torch.ones(len(real_arrs),  dtype=torch.long)
    fake_labels = torch.zeros(len(fake_arrs), dtype=torch.long)

    seqs   = torch.cat([real_t,  fake_t],  dim=0)
    labels = torch.cat([real_labels, fake_labels], dim=0)

    perm = torch.randperm(len(seqs))
    dataset = TensorDataset(seqs[perm], labels[perm])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
