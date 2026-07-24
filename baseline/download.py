"""Fetch and locally cache the ``the-verdict.txt`` corpus.

Unmodified upstream logic from *Build a Large Language Model From Scratch*,
Chapter 2 (Sebastian Raschka). The notebook downloads the public-domain short
story "The Verdict" by Edith Wharton and reads it as the training corpus. The
only addition here is that the download is wrapped in a reusable function with
an explicit local cache path, so the file is fetched at most once.

Source notebook:
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb
"""

from __future__ import annotations

import os

import requests

VERDICT_URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)

DEFAULT_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "the-verdict.txt")


def download_verdict(file_path: str = DEFAULT_CORPUS_PATH) -> str:
    """Download ``the-verdict.txt`` to ``file_path`` if not already cached.

    Mirrors the upstream cell: it only fetches over the network when the file
    is missing, so the corpus is never re-downloaded on subsequent runs.

    Args:
        file_path: Local path to cache the corpus at.

    Returns:
        The path to the (now guaranteed-present) corpus file.
    """
    if not os.path.exists(file_path):
        response = requests.get(VERDICT_URL, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
    return file_path


def load_verdict(file_path: str = DEFAULT_CORPUS_PATH) -> str:
    """Ensure the corpus is cached, then read and return it as text.

    Args:
        file_path: Local path to cache/read the corpus from.

    Returns:
        The full raw text of "The Verdict".
    """
    download_verdict(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
