"""
6VecLM: Word2Vec + Transformer language model for IPv6 target generation.

Paper: "6VecLM: Language Modeling in Vector Space for IPv6 Target Generation"
       Cui et al., ECML PKDD 2020
Reference: https://github.com/CuiTianyu961030/6VecLM

Algorithm
---------
1. **Tokenise** — each nibble is encoded as a 2-char position-aware token
   (nibble_hex + location_alpha[position]), producing up to 512 unique tokens.

2. **Word2Vec** — train a CBOW Word2Vec (gensim 4.x) on the token sentences
   to learn a 100-d embedding space that captures per-nibble co-occurrence
   structure.

3. **Transformer** — train a 6-layer Encoder-Decoder Transformer:
   • Encoder input:  first 16 tokens (network prefix, positions 0–15)
   • Decoder output: last 16 tokens (IID / interface identifier, positions 16–31)
   • Loss:           CosineEmbeddingLoss (predicted vector vs. w2v target vector)
   The embeddings are warm-started from the frozen Word2Vec weights.

4. **Generate** — for each seed × temperature pair, autoregressively decode
   16 IID tokens by temperature-sampling the cosine-similarity distribution
   over 16 candidate tokens per position.

5. **Cache** — both models are serialised to ``data/cache/6veclm/`` together
   with a SHA-256 fingerprint of the seed file.  Subsequent calls with the
   same seeds skip training entirely.

Usage
-----
    from algorithms.six_vec_lm import run
    run(seeds="data/seeds.txt", output="out.txt", budget=1_000_000)
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from algorithms.six_tree.translation import b4_to_std, normalize_to_b4
from .generation import generate_addresses
from .model import make_model
from .preprocessing import build_sentences, train_word2vec


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

CACHE_DIR  = Path("data/cache/6veclm")
_W2V_PATH  = CACHE_DIR / "ipv62vec.model"
_TRANS_PATH = CACHE_DIR / "ipv6_transformer.pt"
_HASH_PATH = CACHE_DIR / "seeds.sha256"

# ---------------------------------------------------------------------------
# Hyperparameters (matching original 6VecLM paper / repo)
# ---------------------------------------------------------------------------

_W2V_VECTOR_SIZE = 100
_W2V_WINDOW      = 5
_W2V_MIN_COUNT   = 1
_W2V_EPOCHS      = 5

_TRANS_N       = 6
_TRANS_D_MODEL = 100   # must equal _W2V_VECTOR_SIZE
_TRANS_D_FF    = 2048
_TRANS_H       = 10
_TRANS_DROPOUT = 0.1
_TRANS_EPOCHS  = 10
_TRANS_BATCH   = 100
_TRANS_LR      = 1e-4

_ENCODER_INPUT_LENGTH = 16   # first 16 nibbles → encoder


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(seeds: str, output: str, budget: int) -> None:
    """Generate IPv6 target addresses using 6VecLM.

    Args:
        seeds:  Path to seed IPv6 address file (one per line).
        output: Path to write generated target addresses.
        budget: Maximum number of addresses to generate.
    """
    # ── Step 1: load seeds ───────────────────────────────────────────────────
    print(f"[6VecLM] Loading seeds from: {seeds}")
    b4_seeds = _load_seeds(seeds)
    if not b4_seeds:
        print("[6VecLM] ERROR: no valid seed addresses — aborting.")
        return
    print(f"[6VecLM] {len(b4_seeds):,} unique seeds loaded.")

    arrs = np.array(
        [[int(c, 16) for c in addr] for addr in b4_seeds], dtype=np.uint8
    )

    # ── Step 2: check cache ──────────────────────────────────────────────────
    seed_hash = _sha256_file(seeds)
    w2v, model = _try_load_cache(seed_hash)

    if w2v is None or model is None:
        # ── Step 3a: train Word2Vec ──────────────────────────────────────────
        print("[6VecLM] Training Word2Vec …")
        sentences = build_sentences(arrs)
        w2v = train_word2vec(
            sentences,
            vector_size=_W2V_VECTOR_SIZE,
            window=_W2V_WINDOW,
            min_count=_W2V_MIN_COUNT,
            epochs=_W2V_EPOCHS,
        )
        print(f"[6VecLM] Word2Vec vocab size: {len(w2v.wv.key_to_index):,}")

        # ── Step 3b: train Transformer ───────────────────────────────────────
        print("[6VecLM] Training Transformer …")
        model = _train_transformer(
            arrs, sentences, w2v,
            epochs=_TRANS_EPOCHS,
            batch_size=_TRANS_BATCH,
            lr=_TRANS_LR,
        )

        # ── Step 3c: save cache ──────────────────────────────────────────────
        _save_cache(w2v, model, seed_hash)
        print(f"[6VecLM] Models cached to: {CACHE_DIR}")
    else:
        print("[6VecLM] Cache hit — skipping training.")

    # ── Step 4: generate ─────────────────────────────────────────────────────
    print(f"[6VecLM] Generating up to {budget:,} target addresses …")
    targets = generate_addresses(
        arrs, model, w2v, budget,
        encoder_input_length=_ENCODER_INPUT_LENGTH,
    )
    print(f"[6VecLM] {len(targets):,} addresses generated.")

    # ── Step 5: write output ─────────────────────────────────────────────────
    with open(output, "w") as fh:
        for addr in targets:
            fh.write(addr + "\n")
    print(f"[6VecLM] Targets written to: {output}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_transformer(
    arrs: np.ndarray,
    sentences: list[list[str]],
    w2v,
    epochs: int = _TRANS_EPOCHS,
    batch_size: int = _TRANS_BATCH,
    lr: float = _TRANS_LR,
):
    """Train the Transformer model on the nibble token sentences.

    Args:
        arrs:       Nibble matrix (n, 32), uint8 (unused; sentences already built).
        sentences:  Token sentences from preprocessing.build_sentences().
        w2v:        Trained gensim Word2Vec model.
        epochs:     Number of training epochs.
        batch_size: Mini-batch size.
        lr:         Adam learning rate.

    Returns:
        Trained EncoderDecoder in eval mode, moved to best available device.
    """
    from .model import subsequent_mask
    from .preprocessing import word2id, id2word

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[6VecLM] Using device: {device}")

    model = make_model(
        w2v,
        N=_TRANS_N,
        d_model=_TRANS_D_MODEL,
        d_ff=_TRANS_D_FF,
        h=_TRANS_H,
        dropout=_TRANS_DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9
    )
    loss_fn = nn.CosineEmbeddingLoss()

    # Precompute w2v vector matrix indexed by token ID (1-based, 0 = zero)
    d_model = w2v.vector_size
    vocab_size = len(w2v.wv.key_to_index)
    w2v_matrix = np.zeros((vocab_size + 1, d_model), dtype=np.float32)
    for word, idx in w2v.wv.key_to_index.items():
        w2v_matrix[idx + 1] = w2v.wv[word]   # 1-based
    w2v_tensor = torch.tensor(w2v_matrix, dtype=torch.float, device=device)

    # Build dataset: (src, tgt_in, tgt_out) per sentence
    #   src     = word IDs for tokens[0:16]
    #   tgt_in  = [BOS=0] + word IDs for tokens[16:31]   (16 tokens, decoder input)
    #   tgt_out = word IDs for tokens[16:32]              (16 tokens, supervision)
    dataset = []
    for sent in sentences:
        src     = [word2id(t, w2v) for t in sent[:_ENCODER_INPUT_LENGTH]]
        iid     = [word2id(t, w2v) for t in sent[_ENCODER_INPUT_LENGTH:]]
        tgt_in  = [0] + iid[:-1]   # BOS + first 15 IID tokens
        tgt_out = iid               # all 16 IID tokens
        dataset.append((src, tgt_in, tgt_out))

    model.train()
    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]

            src     = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
            tgt_in  = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
            tgt_out = torch.tensor([x[2] for x in batch], dtype=torch.long, device=device)

            # Masks
            src_mask = (src != 0).unsqueeze(-2)
            tgt_len  = tgt_in.size(1)
            tgt_mask = (tgt_in != 0).unsqueeze(-2) & subsequent_mask(tgt_len).to(device)

            # Forward
            out         = model(src, tgt_in, src_mask, tgt_mask)  # (B, tgt_len, d_model)
            pred_vectors = model.generator(out)                    # (B, tgt_len, d_model)

            # Look up target Word2Vec vectors
            flat_tgt_out    = tgt_out.view(-1)                     # (B * tgt_len,)
            target_vectors  = w2v_tensor[flat_tgt_out]             # (B * tgt_len, d_model)
            flat_pred       = pred_vectors.view(-1, d_model)       # (B * tgt_len, d_model)

            # Skip padding positions (tgt_out == 0)
            non_pad = flat_tgt_out != 0
            if non_pad.sum() == 0:
                continue

            labels = torch.ones(non_pad.sum(), device=device)
            loss   = loss_fn(flat_pred[non_pad], target_vectors[non_pad], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        print(f"[6VecLM]   Epoch {epoch + 1:2d}/{epochs} — avg loss: {avg:.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_load_cache(seed_hash: str):
    """Load cached models if cache exists and the seed hash matches.

    Returns:
        (w2v, model) on cache hit, (None, None) on miss or hash mismatch.
    """
    if not (_W2V_PATH.exists() and _TRANS_PATH.exists() and _HASH_PATH.exists()):
        return None, None

    cached_hash = _HASH_PATH.read_text().strip()
    if cached_hash != seed_hash:
        print("[6VecLM] Seed file changed — invalidating cache.")
        return None, None

    try:
        from gensim.models import Word2Vec as _W2V
        w2v = _W2V.load(str(_W2V_PATH))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = make_model(w2v).to(device)
        state = torch.load(str(_TRANS_PATH), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        print(f"[6VecLM] Loaded cached models from: {CACHE_DIR}")
        return w2v, model
    except Exception as exc:
        print(f"[6VecLM] Cache load failed ({exc}) — retraining.")
        return None, None


def _save_cache(w2v, model, seed_hash: str) -> None:
    """Save Word2Vec, Transformer state-dict, and seed hash to CACHE_DIR."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    w2v.save(str(_W2V_PATH))
    torch.save(model.state_dict(), str(_TRANS_PATH))
    _HASH_PATH.write_text(seed_hash)


# ---------------------------------------------------------------------------
# Seed loader
# ---------------------------------------------------------------------------

def _load_seeds(path: str) -> list[str]:
    """Read seed file and return sorted unique b4-format addresses."""
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
