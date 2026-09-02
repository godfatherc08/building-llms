"""Baseline tokenizers from Chapter 2.

Unmodified upstream code from *Build a Large Language Model From Scratch*,
Chapter 2 (Sebastian Raschka). Contains:

* :func:`preprocess` -- the regex whitespace/punctuation split used to build a
  word-level vocabulary
* :func:`build_vocab` -- turns a corpus into a ``token -> id`` mapping (and,
  for the V2 variant, appends the ``<|endoftext|>`` and ``<|unk|>`` specials).
* :class:`SimpleTokenizerV1` -- word-level tokenizer that raises ``KeyError``
  on out-of-vocabulary tokens.
* :class:`SimpleTokenizerV2` -- same, but maps unknown tokens to ``<|unk|>``
  and supports the ``<|endoftext|>`` separator.
* :func:`get_bpe_tokenizer` -- the GPT-2 byte-pair-encoding tokenizer via
  ``tiktoken``.

The regex-split and decode logic is copied verbatim from the notebook so this
stays a faithful baseline (including the intentional regex difference between
the ``decode`` methods of V1 and V2).

Source notebook:
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb
"""

from __future__ import annotations

import re
from typing import Dict, List

import tiktoken

_SPLIT_PATTERN = r'([,.:;?_!"()\']|--|\s)'


def preprocess(text: str) -> List[str]:
    """Split ``text`` into word/punctuation tokens, dropping whitespace.

    Args:
        text: Raw input string.

    Returns:
        List of non-empty, stripped tokens.
    """
    preprocessed = re.split(_SPLIT_PATTERN, text)
    return [item.strip() for item in preprocessed if item.strip()]


def build_vocab(text: str, add_special_tokens: bool = False) -> Dict[str, int]:
    """Build a ``token -> id`` vocabulary from a corpus.

    Args:
        text: Corpus to derive the vocabulary from.
        add_special_tokens: If ``True``, append ``<|endoftext|>`` and
            ``<|unk|>`` to the end of the sorted vocabulary (as required by
            :class:`SimpleTokenizerV2`).

    Returns:
        Mapping from token string to integer id.
    """
    all_tokens = sorted(set(preprocess(text)))
    if add_special_tokens:
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    return {token: integer for integer, token in enumerate(all_tokens)}


class SimpleTokenizerV1:
    """Word-level tokenizer with a fixed vocabulary (Section 2.3).

    Raises ``KeyError`` when :meth:`encode` encounters a token that is not in
    the vocabulary -- this is the upstream behavior that motivates V2.
    """

    def __init__(self, vocab: Dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Encode ``text`` into a list of token ids."""
        preprocessed = re.split(_SPLIT_PATTERN, text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back into text."""
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    """Word-level tokenizer with ``<|unk|>`` handling (Section 2.4).

    Unknown tokens are mapped to ``<|unk|>`` instead of raising, and the
    ``<|endoftext|>`` special token is supported for joining independent texts.
    """

    def __init__(self, vocab: Dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Encode ``text``, substituting ``<|unk|>`` for unknown tokens."""
        preprocessed = re.split(_SPLIT_PATTERN, text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back into text."""
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


def get_bpe_tokenizer(encoding_name: str = "gpt2") -> "tiktoken.Encoding":
    """Return the GPT-2 byte-pair-encoding tokenizer (Section 2.5).

    Thin wrapper over ``tiktoken.get_encoding`` so callers don't depend on the
    tiktoken import directly. The returned object exposes ``encode`` /
    ``decode``; pass ``allowed_special={"<|endoftext|>"}`` to ``encode`` when
    the text contains that special token.

    Args:
        encoding_name: Name of the tiktoken encoding (default ``"gpt2"``).

    Returns:
        A ``tiktoken.Encoding`` instance.
    """
    return tiktoken.get_encoding(encoding_name)
