"""Smoke tests for the Chapter 2 baseline.

Verifies that the word-level tokenizers round-trip on sample strings and that
the sliding-window dataloader yields correctly shaped tensors. These are
sanity checks against accidental behavioral drift in the baseline -- not an
exhaustive test suite.
"""

from __future__ import annotations

import torch

from baseline import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocab,
    create_dataloader_v1,
    get_bpe_tokenizer,
)

SAMPLE_CORPUS = (
    "Hello, world. This is a small test corpus. "
    "It has commas, periods; and colons: plus a dash -- here. "
    "The quick brown fox jumps over the lazy dog?"
)


def test_simple_tokenizer_v1_roundtrip() -> None:
    """V1 encodes and decodes a string drawn from its own vocabulary."""
    vocab = build_vocab(SAMPLE_CORPUS)
    tokenizer = SimpleTokenizerV1(vocab)

    text = "Hello, world. This is a test?"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert all(isinstance(i, int) for i in ids)
    assert tokenizer.encode(decoded) == ids


def test_simple_tokenizer_v2_unknown_and_endoftext() -> None:
    """V2 maps out-of-vocabulary tokens to <|unk|> and keeps <|endoftext|>."""
    vocab = build_vocab(SAMPLE_CORPUS, add_special_tokens=True)
    tokenizer = SimpleTokenizerV2(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "This is a test."
    joined = " <|endoftext|> ".join((text1, text2))

    ids = tokenizer.encode(joined)
    decoded = tokenizer.decode(ids)

    assert vocab["<|unk|>"] in ids
    assert vocab["<|endoftext|>"] in ids
    assert "<|endoftext|>" in decoded
    assert "<|unk|>" in decoded


def test_bpe_tokenizer_roundtrip() -> None:
    """The tiktoken GPT-2 tokenizer round-trips text exactly."""
    tokenizer = get_bpe_tokenizer("gpt2")
    text = "Hello, do you like tea? <|endoftext|> Some text."
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    assert tokenizer.decode(ids) == text


def test_dataloader_shapes() -> None:
    """The dataloader yields input/target tensors of shape (batch, max_length)."""
    raw_text = SAMPLE_CORPUS * 20
    batch_size, max_length = 8, 4

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    inputs, targets = next(iter(dataloader))

    assert inputs.shape == (batch_size, max_length)
    assert targets.shape == (batch_size, max_length)
    assert inputs.dtype == torch.int64
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
