"""Runnable walkthrough of the Chapter 2 baseline.

Reproduces the notebook's demonstrations end to end: the word-level tokenizers,
the tiktoken BPE tokenizer, the sliding-window dataloader, and the token +
positional embedding setup. All demo/print code is guarded under ``__main__``
so importing the ``baseline`` package stays side-effect free.

Run from the repository root:

    python -m examples.demo
"""

from __future__ import annotations

import torch

from baseline import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocab,
    compute_input_embeddings,
    create_dataloader_v1,
    create_positional_embedding_layer,
    create_token_embedding_layer,
    get_bpe_tokenizer,
    load_verdict,
)
from baseline.embeddings import DEFAULT_OUTPUT_DIM, GPT2_VOCAB_SIZE


def demo_simple_tokenizers(raw_text: str) -> None:
    """Section 2.3-2.4: word-level tokenizers with and without <|unk|>."""
    print("=" * 60)
    print("SimpleTokenizerV1 / V2")
    print("=" * 60)

    vocab_v1 = build_vocab(raw_text)
    print("V1 vocab size:", len(vocab_v1))
    tokenizer_v1 = SimpleTokenizerV1(vocab_v1)
    sample = (
        '"It\'s the last he painted, you know," '
        "Mrs. Gisburn said with pardonable pride."
    )
    ids = tokenizer_v1.encode(sample)
    print("V1 encode:", ids)
    print("V1 decode:", tokenizer_v1.decode(ids))

    vocab_v2 = build_vocab(raw_text, add_special_tokens=True)
    print("\nV2 vocab size:", len(vocab_v2))
    tokenizer_v2 = SimpleTokenizerV2(vocab_v2)
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    joined = " <|endoftext|> ".join((text1, text2))
    ids = tokenizer_v2.encode(joined)
    print("V2 encode:", ids)
    print("V2 decode:", tokenizer_v2.decode(ids))


def demo_bpe() -> None:
    """Section 2.5: GPT-2 byte-pair encoding via tiktoken."""
    print("\n" + "=" * 60)
    print("BPE (tiktoken gpt2)")
    print("=" * 60)

    tokenizer = get_bpe_tokenizer("gpt2")
    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace ."
    )
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print("BPE encode:", integers)
    print("BPE decode:", tokenizer.decode(integers))


def demo_dataloader(raw_text: str) -> None:
    """Section 2.6: sliding-window input/target pairs."""
    print("\n" + "=" * 60)
    print("Dataloader (sliding window)")
    print("=" * 60)

    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    inputs, targets = next(iter(dataloader))
    print("Inputs shape:", inputs.shape)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)


def demo_embeddings(raw_text: str) -> None:
    """Section 2.7-2.8: token + positional embeddings."""
    print("\n" + "=" * 60)
    print("Token + positional embeddings")
    print("=" * 60)

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False,
    )
    inputs, _ = next(iter(dataloader))

    token_embedding_layer = create_token_embedding_layer(
        GPT2_VOCAB_SIZE, DEFAULT_OUTPUT_DIM
    )
    pos_embedding_layer = create_positional_embedding_layer(
        max_length, DEFAULT_OUTPUT_DIM
    )

    token_embeddings = token_embedding_layer(inputs)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    input_embeddings = compute_input_embeddings(
        inputs, token_embedding_layer, pos_embedding_layer
    )

    print("Token embeddings shape:  ", tuple(token_embeddings.shape))
    print("Positional embeddings:   ", tuple(pos_embeddings.shape))
    print("Input embeddings shape:  ", tuple(input_embeddings.shape))


def main() -> None:
    raw_text = load_verdict()
    print("Total number of characters:", len(raw_text))
    print(raw_text[:99])
    demo_simple_tokenizers(raw_text)
    demo_bpe()
    demo_dataloader(raw_text)
    demo_embeddings(raw_text)


if __name__ == "__main__":
    main()
