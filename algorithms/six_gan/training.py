"""
6GAN training: MLE pre-training → discriminator pre-training → SeqGAN adversarial.

Ported from CuiTianyu961030/6GAN (train.py), rewritten for PyTorch.

Three-phase training schedule (matching the original paper)
-----------------------------------------------------------
Phase 1 – Generator MLE pre-training (50 epochs)
  Maximise log P(real_nibbles | BOS, prev_nibbles) via cross-entropy.
  This gives the generator a reasonable starting point before GAN training.

Phase 2 – Discriminator pre-training (10 iterations)
  Train the CNN discriminator to separate real seeds from generator samples.

Phase 3 – Adversarial training (SeqGAN, default 200 mini-batches)
  Generator update: REINFORCE with per-step Monte Carlo rewards.
    - For each time step t, rollout_num completions of the partial
      sequence are scored by the discriminator (P(real)).
    - The average score is the reward R_t for that step.
    - Loss = −Σ R_t · log π_θ(a_t | s_t)
  Discriminator update: MLE on fresh real + generated batches (d_steps=3).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import BOS_TOKEN, SEQ_LEN, make_dis_dataloader, make_gen_dataloader
from .model import CNNDiscriminator, LSTMGenerator


# ---------------------------------------------------------------------------
# Phase 1: Generator MLE pre-training
# ---------------------------------------------------------------------------

def mle_pretrain(
    generator: LSTMGenerator,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.01,
) -> None:
    """Pre-train generator with teacher-forcing cross-entropy loss.

    Args:
        generator:  LSTMGenerator to train (modified in-place).
        dataloader: Yields (inp, tgt) batches from make_gen_dataloader().
        device:     Torch device.
        epochs:     Number of full passes over the dataset.
        lr:         Adam learning rate.
    """
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    generator.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches  = 0

        for inp, tgt in dataloader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits, _ = generator(inp)                          # (B, seq, vocab)
            loss = F.cross_entropy(
                logits.reshape(-1, generator.vocab_size),
                tgt.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[6GAN] MLE Epoch {epoch + 1:3d}/{epochs} — loss: {avg:.4f}")


# ---------------------------------------------------------------------------
# Phase 2: Discriminator pre-training
# ---------------------------------------------------------------------------

def discriminator_pretrain(
    discriminator: CNNDiscriminator,
    generator: LSTMGenerator,
    real_arrs: np.ndarray,
    device: torch.device,
    iterations: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
) -> None:
    """Pre-train discriminator on real seeds vs. MLE-generated sequences.

    Args:
        discriminator: CNNDiscriminator to train (modified in-place).
        generator:     Trained (or pre-trained) generator for negative samples.
        real_arrs:     (n, 32) nibble matrix of real seed addresses.
        device:        Torch device.
        iterations:    Number of pre-training sweeps.
        batch_size:    Mini-batch size.
        lr:            Adam learning rate.
    """
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    for it in range(iterations):
        n_fake    = min(len(real_arrs), 10_000)
        fake_arrs = generator.sample(n_fake, device).cpu().numpy()
        loader    = make_dis_dataloader(real_arrs, fake_arrs, batch_size)

        discriminator.train()
        total_loss = 0.0
        n_batches  = 0

        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            loss = F.cross_entropy(discriminator(seqs), labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        print(
            f"[6GAN] Discriminator pre-train {it + 1:2d}/{iterations}"
            f" — loss: {avg:.4f}"
        )


# ---------------------------------------------------------------------------
# Phase 3: Adversarial training (SeqGAN)
# ---------------------------------------------------------------------------

def adversarial_train(
    generator: LSTMGenerator,
    discriminator: CNNDiscriminator,
    real_arrs: np.ndarray,
    device: torch.device,
    total_batches: int = 200,
    batch_size: int = 64,
    rollout_num: int = 16,
    g_lr: float = 1e-4,
    d_lr: float = 1e-4,
    d_steps: int = 3,
    log_interval: int = 20,
) -> None:
    """SeqGAN adversarial training.

    Generator update — REINFORCE:
      For each generated sequence, compute per-step rewards via Monte Carlo
      rollout: at step t, rollout_num independent completions of the partial
      sequence (tokens 0…t) are drawn from the generator and scored by the
      discriminator P(real).  The average score is the reward R_t.

      REINFORCE loss = −Σ_t R_t · log π_θ(a_t | s_{<t})

    Discriminator update — MLE:
      After each generator step, run d_steps mini-batches of real + fresh
      generated sequences through the discriminator with cross-entropy loss.

    Args:
        generator:     LSTMGenerator to train.
        discriminator: CNNDiscriminator to train.
        real_arrs:     (n, 32) nibble matrix of real seed addresses.
        device:        Torch device.
        total_batches: Number of adversarial generator update steps.
        batch_size:    Mini-batch size.
        rollout_num:   Monte Carlo completions per (sequence, timestep).
        g_lr:          Generator Adam learning rate.
        d_lr:          Discriminator Adam learning rate.
        d_steps:       Discriminator update steps per generator step.
        log_interval:  Print progress every this many batches.
    """
    g_opt = torch.optim.Adam(generator.parameters(),     lr=g_lr)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=d_lr)

    for batch_idx in range(total_batches):

        # ── Generator update (REINFORCE) ──────────────────────────────────
        generator.train()
        discriminator.eval()

        inp    = torch.full((batch_size, 1), BOS_TOKEN, dtype=torch.long, device=device)
        hidden = None
        generated_toks:  list[torch.Tensor] = []
        log_probs_list:  list[torch.Tensor] = []

        for _ in range(SEQ_LEN):
            logits, hidden = generator(inp, hidden)          # (B, 1, vocab)
            logits = logits[:, -1, :]                        # (B, vocab)
            logits[:, BOS_TOKEN:] = -1e9                     # mask special tokens
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)           # (B, 1)
            log_prob = torch.log(probs.gather(1, next_tok) + 1e-8)
            generated_toks.append(next_tok)
            log_probs_list.append(log_prob)
            inp = next_tok

        gen_seqs  = torch.cat(generated_toks,  dim=1)       # (B, seq_len)
        log_probs = torch.cat(log_probs_list,  dim=1)       # (B, seq_len)

        # Monte Carlo rewards
        rewards = _mc_rewards(generator, discriminator, gen_seqs, rollout_num, device)

        # REINFORCE: −E[R · log π]
        g_loss = -(rewards * log_probs).mean()
        g_opt.zero_grad()
        g_loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
        g_opt.step()

        # ── Discriminator update (MLE) ─────────────────────────────────────
        for _ in range(d_steps):
            n_fake    = min(len(real_arrs), batch_size * 2)
            fake_arrs = generator.sample(n_fake, device).cpu().numpy()
            loader    = make_dis_dataloader(real_arrs[:n_fake], fake_arrs, batch_size)
            discriminator.train()

            for seqs, labels in loader:
                seqs, labels = seqs.to(device), labels.to(device)
                d_loss = F.cross_entropy(discriminator(seqs), labels)
                d_opt.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
                d_opt.step()

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"[6GAN] Adversarial {batch_idx + 1:4d}/{total_batches}"
                f" — g_loss: {g_loss.item():.4f}"
            )


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def _mc_rewards(
    generator: LSTMGenerator,
    discriminator: CNNDiscriminator,
    gen_seqs: torch.Tensor,
    rollout_num: int,
    device: torch.device,
) -> torch.Tensor:
    """Estimate per-step rewards via Monte Carlo rollout + discriminator.

    At each time step t (0-indexed):
      • Complete gen_seqs[:, :t+1] to full length *rollout_num* times.
      • Score each completion: P(real) = softmax(discriminator(seq))[:,1].
      • Average over rollouts → R_t for each sequence in the batch.

    At the final step (t = seq_len−1):
      • Score gen_seqs directly (no rollout required).

    Returns:
        (batch, seq_len) reward tensor (values in [0, 1]).
    """
    batch   = gen_seqs.size(0)
    seq_len = gen_seqs.size(1)
    rewards = torch.zeros(batch, seq_len, device=device)

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for t in range(seq_len - 1):
            partial    = gen_seqs[:, :t + 1]               # (B, t+1)
            completed  = generator.rollout_complete(partial, rollout_num, device)
            # (B * rollout_num, seq_len)
            p_real     = F.softmax(discriminator(completed), dim=-1)[:, 1]
            # (B * rollout_num,) → (B, rollout_num) → average over rollouts
            rewards[:, t] = p_real.view(batch, rollout_num).mean(dim=1)

        # Final position: score the complete sequence directly
        p_real_final  = F.softmax(discriminator(gen_seqs), dim=-1)[:, 1]
        rewards[:, -1] = p_real_final

    return rewards
