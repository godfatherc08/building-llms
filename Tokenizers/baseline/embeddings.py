"""Token and positional embedding setup from Chapter 2 (Sections 2.7-2.8).

Unmodified upstream code from *Build a Large Language Model From Scratch*,
Chapter 2 (Sebastian Raschka). Wraps the chapter's closing example -- a GPT-2
sized token embedding layer plus an absolute positional embedding layer, summed
into the final input embeddings -- into small reusable helpers.

The GPT-2 defaults used throughout the chapter are a 50,257-token vocabulary
and a 256-dimensional embedding.

Source notebook:
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb
"""

from __future__ import annotations

import torch

GPT2_VOCAB_SIZE = 50257
DEFAULT_OUTPUT_DIM = 256


def create_token_embedding_layer(
    vocab_size: int = GPT2_VOCAB_SIZE,
    output_dim: int = DEFAULT_OUTPUT_DIM,
) -> torch.nn.Embedding:
    """Create the token embedding layer (Section 2.7).

    Args:
        vocab_size: Number of tokens in the vocabulary.
        output_dim: Embedding dimensionality.

    Returns:
        An ``nn.Embedding`` of shape ``(vocab_size, output_dim)``.
    """
    return torch.nn.Embedding(vocab_size, output_dim)


def create_positional_embedding_layer(
    context_length: int,
    output_dim: int = DEFAULT_OUTPUT_DIM,
) -> torch.nn.Embedding:
    """Create the absolute positional embedding layer (Section 2.8).

    Args:
        context_length: Maximum sequence length (number of positions).
        output_dim: Embedding dimensionality; must match the token embedding.

    Returns:
        An ``nn.Embedding`` of shape ``(context_length, output_dim)``.
    """
    return torch.nn.Embedding(context_length, output_dim)


def compute_input_embeddings(
    token_ids: torch.Tensor,
    token_embedding_layer: torch.nn.Embedding,
    pos_embedding_layer: torch.nn.Embedding,
) -> torch.Tensor:
    """Combine token and positional embeddings into input embeddings.

    Mirrors the chapter's final step: look up token embeddings for a batch of
    ids, look up positional embeddings for positions ``0..context_length-1``,
    and add them (broadcasting the positional embeddings across the batch).

    Args:
        token_ids: Batch of token ids, shape ``(batch_size, context_length)``.
        token_embedding_layer: Layer mapping ids to token embeddings.
        pos_embedding_layer: Layer mapping positions to positional embeddings.

    Returns:
        Input embeddings of shape ``(batch_size, context_length, output_dim)``.
    """
    context_length = token_ids.shape[1]
    token_embeddings = token_embedding_layer(token_ids)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    return token_embeddings + pos_embeddings
