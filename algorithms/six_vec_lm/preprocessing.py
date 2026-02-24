"""
6VecLM preprocessing: tokenize IPv6 addresses and train Word2Vec.

Each nibble is encoded as a 2-character position-aware token:
    token = nibble_hex_char + LOCATION_ALPHA[position]

Examples: '20' (nibble 2 at pos 0), '1a' (nibble 1 at pos 10), 'fb' (nibble f at pos 11).
Vocabulary: up to 16 × 32 = 512 unique tokens.
"""

from __future__ import annotations

import numpy as np
from gensim.models import Word2Vec

# 32-character base-32 alphabet for position encoding
LOCATION_ALPHA = '0123456789abcdefghijklmnopqrstuv'
HEX_CHARS = '0123456789abcdef'


def tokenize_address(b4: str) -> list[str]:
    """Convert a 32-char b4 address to 32 position-aware tokens.

    Args:
        b4: 32-char lowercase hex string (nibble format).

    Returns:
        List of 32 tokens, each 2 chars: nibble_char + position_char.
    """
    return [b4[i] + LOCATION_ALPHA[i] for i in range(32)]


def build_sentences(arrs: np.ndarray) -> list[list[str]]:
    """Convert nibble matrix to a list of token sentences.

    Args:
        arrs: uint8 numpy array of shape (n, 32), each cell 0–15.

    Returns:
        List of n sentences; each sentence is 32 position-aware token strings.
    """
    sentences: list[list[str]] = []
    for row in arrs:
        tokens = [HEX_CHARS[row[i]] + LOCATION_ALPHA[i] for i in range(32)]
        sentences.append(tokens)
    return sentences


def train_word2vec(
    sentences: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    epochs: int = 5,
) -> Word2Vec:
    """Train a Word2Vec model on tokenized IPv6 sentences (gensim 4.x API).

    Args:
        sentences:   List of token sentences from build_sentences().
        vector_size: Embedding dimensionality (must match Transformer d_model).
        window:      Context window size.
        min_count:   Minimum token frequency to include in vocabulary.
        epochs:      Number of training passes.

    Returns:
        Trained gensim Word2Vec model.
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0,       # CBOW
        epochs=epochs,
        workers=4,
    )
    return model


def word2id(word: str, model: Word2Vec) -> int:
    """Return the 1-based integer ID for a token (0 = padding / unknown).

    Uses model.wv.key_to_index (gensim ≥ 4.0 API).
    """
    kti = model.wv.key_to_index
    if word in kti:
        return kti[word] + 1   # 1-based
    return 0                   # unknown / padding


def id2word(idx: int, model: Word2Vec) -> str:
    """Return the token string for a 1-based integer ID (empty string for 0).

    Uses model.wv.index_to_key (gensim ≥ 4.0 API).
    """
    itk = model.wv.index_to_key
    if idx <= 0 or idx > len(itk):
        return ''
    return itk[idx - 1]
