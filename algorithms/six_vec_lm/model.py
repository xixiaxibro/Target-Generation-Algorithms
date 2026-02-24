"""
6VecLM Transformer model.

Ported from CuiTianyu961030/6VecLM (ipv6_transformer.py).

Architecture: standard Encoder-Decoder Transformer (Vaswani et al., 2017).
  - Encoder input:  first 16 nibble tokens (IPv6 prefix / network ID)
  - Decoder output: autoregressive generation of last 16 nibble tokens (IID)
  - Generator:      Linear(d_model → d_model) + Sigmoid (embedding-space output)
  - Loss:           CosineEmbeddingLoss (predicted vector vs. w2v target vector)

API fixes vs. original:
  - nn.init.xavier_uniform_(p)   (inplace; was xavier_uniform without underscore)
  - Variable() removed            (PyTorch ≥ 1.0 handles tensors directly)
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def subsequent_mask(size: int) -> torch.BoolTensor:
    """Return a lower-triangular boolean mask (True = allowed to attend).

    Shape: (size, size). Prevents each position from attending to future ones.
    """
    return ~torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)


def _clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Return N deep copies of a module as a ModuleList."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class EncoderDecoder(nn.Module):
    """Standard Encoder-Decoder Transformer."""

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Linear(d_model → d_model) + Sigmoid.

    Outputs a vector in embedding space; compared to Word2Vec target vectors
    via CosineEmbeddingLoss rather than a softmax over vocabulary logits.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(x))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, layer: nn.Module, N: int):
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn, feed_forward, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = _clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, layer: nn.Module, N: int):
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size: int, self_attn, src_attn, feed_forward, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = _clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = _clones(nn.Linear(d_model, d_model), 4)
        self.attn: torch.Tensor | None = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Linear projections in batch → (batch, h, seq_len, d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Scaled dot-product attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) Concatenate and final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# ---------------------------------------------------------------------------
# Feed-forward and positional encoding
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Embedding with Word2Vec initialisation
# ---------------------------------------------------------------------------

class Embeddings(nn.Module):
    """Token embedding layer, warm-started from Word2Vec vectors.

    Token IDs are 1-based (0 = padding/BOS). The embedding weight at index i
    is initialised from the (i-1)-th Word2Vec vector.
    """

    def __init__(self, d_model: int, vocab_size: int, w2v_matrix=None):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if w2v_matrix is not None:
            n_rows = min(w2v_matrix.shape[0], vocab_size - 1)
            with torch.no_grad():
                self.lut.weight[1:n_rows + 1] = torch.tensor(
                    w2v_matrix[:n_rows], dtype=torch.float
                )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


# ---------------------------------------------------------------------------
# Sublayer helpers
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Add & Norm wrapper (pre-norm variant: norm → sublayer → add residual)."""

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(
    word2vec_model: Word2Vec,
    N: int = 6,
    d_model: int = 100,
    d_ff: int = 2048,
    h: int = 10,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """Build and initialise an EncoderDecoder Transformer.

    Args:
        word2vec_model: Trained gensim Word2Vec; its vectors warm-start the
                        shared encoder/decoder embedding matrices.
        N:       Number of encoder/decoder layers.
        d_model: Model dimensionality (must equal Word2Vec vector_size).
        d_ff:    Feed-forward inner dimensionality.
        h:       Number of attention heads.
        dropout: Dropout probability.

    Returns:
        EncoderDecoder with Xavier-initialised parameters (embeddings
        overwritten with Word2Vec weights).
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # Vocab size: one slot per unique w2v token (1-based) + slot 0 for pad/BOS
    vocab_size = len(word2vec_model.wv.key_to_index) + 1
    w2v_vectors = word2vec_model.wv.vectors   # shape (vocab, d_model)

    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed=nn.Sequential(
            Embeddings(d_model, vocab_size, w2v_vectors), c(position)
        ),
        tgt_embed=nn.Sequential(
            Embeddings(d_model, vocab_size, w2v_vectors), c(position)
        ),
        generator=Generator(d_model),
    )

    # Xavier initialisation for all weight tensors (dim ≥ 2)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
