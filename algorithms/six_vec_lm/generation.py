"""
6VecLM generation: greedy/temperature decode + address reconstruction.

At each decoder step the Generator outputs an embedding-space vector.
next_generation() selects the next nibble token by computing cosine
similarity between that vector and all 16 candidate token embeddings at
the current position, then temperature-sampling the softmax distribution.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from gensim.models import Word2Vec

from algorithms.six_tree.translation import b4_to_std
from .model import EncoderDecoder, subsequent_mask
from .preprocessing import (
    HEX_CHARS,
    LOCATION_ALPHA,
    build_sentences,
    id2word,
    word2id,
)


def next_generation(
    w2v: Word2Vec,
    vector: np.ndarray,
    temperature: float,
    position_idx: int,
) -> int:
    """Select the next nibble token via temperature-scaled cosine similarity.

    For the given address position (16–31) and all 16 possible nibble values,
    computes the cosine similarity between the predicted embedding vector and
    each candidate's Word2Vec vector, then draws a sample from the resulting
    temperature-softmax distribution.

    Args:
        w2v:          Trained gensim Word2Vec model.
        vector:       Predicted embedding vector from model.generator, shape (d_model,).
        temperature:  Sampling temperature; lower ≈ more deterministic.
                      Typical range: 0.015 (near-greedy) to 0.5 (exploratory).
        position_idx: Nibble position in [16, 31].

    Returns:
        1-based token ID of the selected nibble token (or 0 on fallback).
    """
    kti = w2v.wv.key_to_index
    pos_char = LOCATION_ALPHA[position_idx]

    candidates: list[str] = []
    vecs: list[np.ndarray] = []

    for nibble_val in range(16):
        token = HEX_CHARS[nibble_val] + pos_char
        if token in kti:
            candidates.append(token)
            vecs.append(w2v.wv[token])

    if not candidates:
        # Fallback: return a random valid token id at this position
        fallback_tok = HEX_CHARS[random.randint(0, 15)] + pos_char
        return kti.get(fallback_tok, 0) + 1

    # Cosine similarities
    vec_norm = vector / (np.linalg.norm(vector) + 1e-8)
    scores = np.array(
        [np.dot(vec_norm, v / (np.linalg.norm(v) + 1e-8)) for v in vecs],
        dtype=np.float64,
    )

    # Temperature-scaled softmax (subtract max for numerical stability)
    scores = scores / temperature
    scores -= scores.max()
    probs = np.exp(scores)
    probs /= probs.sum()

    chosen_idx: int = np.random.choice(len(candidates), p=probs)
    chosen_word = candidates[chosen_idx]
    return kti[chosen_word] + 1   # 1-based


def greedy_decode(
    model: EncoderDecoder,
    w2v: Word2Vec,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int = 16,
    start_symbol: int = 0,
    temperature: float = 0.015,
) -> torch.Tensor:
    """Autoregressively decode IID nibble tokens (positions 16–31).

    Args:
        model:        Trained EncoderDecoder.
        w2v:          Trained Word2Vec (used by next_generation).
        src:          Encoder input tokens, shape (1, encoder_input_length).
        src_mask:     Encoder attention mask, shape (1, 1, encoder_input_length).
        max_len:      Number of tokens to generate (16 for a full IID).
        start_symbol: BOS token id (0 = padding slot).
        temperature:  Sampling temperature forwarded to next_generation.

    Returns:
        Tensor of shape (1, max_len) with generated token IDs (1-based).
    """
    device = src.device
    memory = model.encode(src, src_mask)

    # Decoder input starts with BOS
    ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

    for step in range(max_len):
        tgt_mask = subsequent_mask(ys.size(1)).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)   # (1, seq, d_model)
        pred_vec = model.generator(out[:, -1])               # (1, d_model)
        vec_np = pred_vec.detach().cpu().numpy()[0]          # (d_model,)

        next_id = next_generation(
            w2v, vec_np, temperature, position_idx=16 + step
        )
        next_token = torch.full((1, 1), next_id, dtype=torch.long, device=device)
        ys = torch.cat([ys, next_token], dim=1)

    return ys[:, 1:]   # strip BOS → (1, max_len)


def generate_addresses(
    arrs: np.ndarray,
    model: EncoderDecoder,
    w2v: Word2Vec,
    budget: int,
    encoder_input_length: int = 16,
    temperatures: tuple[float, ...] = (0.015, 0.02, 0.05, 0.1, 0.2, 0.5),
) -> list[str]:
    """Generate up to *budget* IPv6 target addresses.

    Strategy: iterate (seed × temperature) pairs.  Each pair produces one
    candidate address (prefix from seed, IID from decoder).  If
    seeds × temperatures < budget, loop seeds again with a fresh random
    state until the budget is reached or a safety cap is hit.

    Args:
        arrs:                 Nibble matrix (n, 32), uint8.
        model:                Trained EncoderDecoder (eval mode expected).
        w2v:                  Trained Word2Vec.
        budget:               Maximum number of unique addresses to return.
        encoder_input_length: Number of prefix nibbles fed to encoder (16).
        temperatures:         Temperature values to cycle through per seed.

    Returns:
        List of standard IPv6 address strings (up to *budget* unique entries).
    """
    device = next(model.parameters()).device
    model.eval()

    sentences = build_sentences(arrs)
    n_seeds = len(sentences)
    results: set[str] = set()

    # Safety cap: avoid infinite loop when seeds are very few
    max_passes = max(math.ceil(budget / max(n_seeds * len(temperatures), 1)) + 2, 10)
    pass_num = 0

    with torch.no_grad():
        while len(results) < budget and pass_num < max_passes:
            for sent in sentences:
                if len(results) >= budget:
                    break

                # Build encoder input
                src_ids = [word2id(t, w2v) for t in sent[:encoder_input_length]]
                src = torch.tensor([src_ids], dtype=torch.long, device=device)
                src_mask = (src != 0).unsqueeze(-2)

                # Extract prefix nibbles from seed
                prefix_nibbles = [int(t[0], 16) for t in sent[:encoder_input_length]]

                for temp in temperatures:
                    if len(results) >= budget:
                        break

                    decoded = greedy_decode(
                        model, w2v, src, src_mask,
                        max_len=16, start_symbol=0, temperature=temp,
                    )   # (1, 16)

                    # Decode token IDs → nibble values
                    suffix_nibbles: list[int] = []
                    for tok_id in decoded[0].tolist():
                        word = id2word(tok_id, w2v)
                        if word:
                            suffix_nibbles.append(int(word[0], 16))
                        else:
                            suffix_nibbles.append(random.randint(0, 15))

                    # Reconstruct full 32-nibble address
                    nibbles = prefix_nibbles + suffix_nibbles
                    b4 = ''.join(HEX_CHARS[n] for n in nibbles)
                    std = b4_to_std(b4)
                    results.add(std)

            pass_num += 1

    return list(results)[:budget]
