"""Sliding-window dataset and dataloader from Chapter 2 (Section 2.6).

Unmodified upstream code from *Build a Large Language Model From Scratch*,
Chapter 2 (Sebastian Raschka). Builds ``(input, target)`` pairs where the
target is the input shifted right by one position, using a sliding window over
the BPE-encoded corpus.

Source notebook:
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb
"""

from __future__ import annotations

from typing import Tuple

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    """Chunk a text into overlapping ``(input, target)`` id sequences.

    Args:
        txt: Raw corpus text.
        tokenizer: A tiktoken encoding (or compatible) with an ``encode``
            method accepting ``allowed_special``.
        max_length: Context length of each chunk.
        stride: Step size between the start of consecutive chunks.
    """

    def __init__(
        self,
        txt: str,
        tokenizer: "tiktoken.Encoding",
        max_length: int,
        stride: int,
    ) -> None:
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a ``DataLoader`` of sliding-window ``(input, target)`` batches.

    Args:
        txt: Raw corpus text.
        batch_size: Number of sequences per batch.
        max_length: Context length of each sequence.
        stride: Step size between the start of consecutive sequences.
        shuffle: Whether to shuffle the sequences each epoch.
        drop_last: Drop the last partial batch if it is smaller than
            ``batch_size``.
        num_workers: Number of subprocesses used for data loading.

    Returns:
        A configured ``torch.utils.data.DataLoader``.
    """

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
