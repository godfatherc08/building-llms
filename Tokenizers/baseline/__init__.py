"""Baseline (unmodified) tokenization code ported from Chapter 2 of
*Build a Large Language Model From Scratch* by Sebastian Raschka.

This package is a faithful port of the chapter's reference implementation. It
is intentionally kept identical to upstream so that later modifications (in
sibling packages) diff cleanly against a known-good baseline. See README.md in
this directory for attribution and license.
"""

from __future__ import annotations

from Tokenizers.baseline.dataset import GPTDatasetV1, create_dataloader_v1
from Tokenizers.baseline.download import DEFAULT_CORPUS_PATH, download_verdict, load_verdict
from Tokenizers.baseline.embeddings import (
    compute_input_embeddings,
    create_positional_embedding_layer,
    create_token_embedding_layer,
)
from Tokenizers.baseline.tokenizers import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocab,
    get_bpe_tokenizer,
    preprocess,
)

__all__ = [
    "GPTDatasetV1",
    "create_dataloader_v1",
    "DEFAULT_CORPUS_PATH",
    "download_verdict",
    "load_verdict",
    "compute_input_embeddings",
    "create_positional_embedding_layer",
    "create_token_embedding_layer",
    "SimpleTokenizerV1",
    "SimpleTokenizerV2",
    "build_vocab",
    "get_bpe_tokenizer",
    "preprocess",
]
