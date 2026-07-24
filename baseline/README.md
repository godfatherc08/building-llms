# Baseline: Chapter 2 tokenization (unmodified upstream)

This package is a **faithful, unmodified port** of the reference tokenization
code from **Chapter 2 ("Working with Text Data")** of
[*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch)
by **Sebastian Raschka**.

It exists to serve as a clean baseline: our modifications (unknown-word
handling, multilingual fertility, numerical tokenization, glitch tokens, …)
live in **sibling packages**, so the diff against *what upstream does* stays
legible. **Do not add feature changes here.** If behavior needs to change,
copy the relevant piece into a new package and modify it there.

## Attribution & license

- **Source repository:** https://github.com/rasbt/LLMs-from-scratch
- **Source notebook:** [`ch02/01_main-chapter-code/ch02.ipynb`](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb)
- **Author:** Sebastian Raschka
- **Upstream license:** Apache License 2.0 (see the
  [LICENSE](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt)
  in the source repository). The code here is derived from that repository and
  remains under its terms.
- **Corpus:** *The Verdict* by Edith Wharton — public domain
  ([Wikisource](https://en.wikisource.org/wiki/The_Verdict)).

## What was ported

| Upstream (notebook) | Here |
| --- | --- |
| `the-verdict.txt` download cell | `download.py` (`download_verdict`, `load_verdict`) — cached locally |
| `SimpleTokenizerV1`, `SimpleTokenizerV2`, vocab build, BPE | `tokenizers.py` |
| `GPTDatasetV1`, `create_dataloader_v1` | `dataset.py` |
| Token + positional embedding cells | `embeddings.py` |
| Notebook demo cells | `../examples/demo.py` |

## What was changed (structure only, not behavior)

- Notebook cells were split into modules by concern; `%` magics, bare display
  expressions, and inline prose were dropped.
- Type hints and short docstrings were added.
- The corpus download is wrapped in a function with an explicit local cache
  path so it is fetched at most once.
- All demo/print code is guarded under `examples/demo.py` (`__main__`), so
  importing `baseline` is side-effect free.

The **core token-producing logic is byte-for-byte the upstream logic.** One
upstream quirk was preserved deliberately: the `decode` regex differs between
`SimpleTokenizerV1` (omits `:` and `;`) and `SimpleTokenizerV2` (includes them).

## Pinned versions

`tiktoken==0.7.0`, `torch==2.5.1` — matching the upstream notebook so BPE token
ids reproduce exactly. See `../requirements.txt`.

## Running

```bash
pip install -r ../requirements.txt

# End-to-end walkthrough (downloads + caches the corpus on first run):
python -m examples.demo

# Smoke tests:
pytest tests/
```

### Note for TLS-intercepting networks

On networks that intercept TLS (corporate proxies with their own root CA),
Python's `requests`/`tiktoken` downloads fail with
`CERTIFICATE_VERIFY_FAILED` even though the OS trusts the proxy root. Two
caches make everything run offline once populated:

- `baseline/the-verdict.txt` — the corpus (fetched once by `download.py`, or
  drop it in manually).
- A tiktoken vocab cache. Pre-download `vocab.bpe` and `encoder.json` (named by
  the sha1 of their blob URLs) into a directory and point tiktoken at it:

  ```bash
  export TIKTOKEN_CACHE_DIR=/path/to/.tiktoken_cache
  ```

This working copy already has a populated `.tiktoken_cache/` at the repo root
(git-ignored, not committed); set `TIKTOKEN_CACHE_DIR` to its path to run the
BPE demo/tests without network.
