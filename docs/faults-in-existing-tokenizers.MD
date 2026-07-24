# Faults in Existing Numerical Tokenization Schemes

Three baselines: left-to-right fixed chunking (the GPT-3.5/4 default), right-to-left fixed chunking (Singh & Strouse, 2024), and single-digit tokenization (the Llama/PaLM default). Value-aware embeddings (Dutulescu et al., 2026) sit at the embedding layer rather than the tokenizer, so they're out of scope here. The place-value scheme is covered in its own document.

---

## 1. Left-to-right fixed-width chunking

BPE chunks greedily from the left, up to three digits at a time. `9876543` becomes `987|654|3`.

The failure is alignment. Chunk boundaries in the sum are computed independently of the boundaries in the addends, so when a final carry adds a digit they land somewhere else entirely. `9876543 + 8765432` chunks as `987|654|3` and `876|543|2`; the sum, `18641975`, chunks as `186|419|75`. Nothing in the sum corresponds to anything in either addend.

Right-to-left chunking beats this by up to 20 percentage points on 7–9 digit addition. Restrict to problems where a final carry adds a digit and accuracy under L2R drops to roughly 8%. That collapse doesn't track how many carries the problem contains, which rules out "harder problems fail more" — the boundary shift is doing the damage on its own.

Two things about the errors are worth holding onto. They're stereotyped: the same digit position goes wrong regardless of what belonged there, which is the signature of a fixed but wrong rule rather than noisy computation. And the residual scattered errors in both L2R and R2L cluster at token boundaries, so the seam between tokens is a weak point generally, not only under length mismatch.

Scale doesn't resolve it. The effect shrinks from GPT-3.5 to GPT-4 and returns in the smaller distilled GPT-4 Turbo, which suggests a larger model partially overrides the bias rather than removing its cause.

Underneath all of this: the chunking rule is an artifact of frequency-driven merges over training text. It has no idea the substrings it's merging carry place value. The same merge logic on a word would be unremarkable.

---

## 2. Right-to-left fixed-width chunking

Identical logic, anchored at the ones digit instead of the start of the string. `9876543` becomes `9|876|543`.

Because grouping always starts from the same reference point, boundaries stop moving when a carry changes the overall length, and the L2R failure goes away.

What doesn't go away: the grouping is still three characters wide and still has no concept of what a hundred or a thousand is — it's blind counting, just counting from the other end. Token count is identical to L2R, so there's no compression to show for it. `543` and `544` remain unrelated identities in the embedding table. And it has only been demonstrated as a workaround, by forcing the behaviour with inserted commas, not as a retrained tokenizer inside a full pretraining pipeline; whether it survives that unchanged is untested. No frontier model currently ships it as the default.

---

## 3. Single-digit tokenization

`9876543` becomes `9|8|7|6|5|4|3`. Misalignment can't occur because there's no grouping left to misalign, and of the pure tokenization choices this generalizes best in-domain and out-of-domain across every model size tested.

The cost is that place value becomes entirely positional. A digit's magnitude is now just a function of how far into the number the model currently is. Isolated digit-position tasks — return the 5th digit, nothing else — fall to around 20% accuracy on long numbers, against a 10% random-guess floor. The scheme leans its whole weight on the one skill these models are demonstrably worst at.

Positional encoding compounds this. RoPE, the default nearly everywhere, tends to memorize training-time lengths rather than generalize. Alibi and NoPE do better but are standard almost nowhere, so a second length-generalization weakness stacks on the first.

The rest of the costs are mechanical. Sequence length goes up roughly 3x against 3-digit chunking: more compute, more context consumed per number, one full decode step per digit. `1,000,000` costs exactly as much as a random seven-digit number, since no redundancy is ever exploited. And it can't be retrofitted — grafting a new tokenizer onto an already-pretrained model backfired against behaviour the model had already learned, so this is a pretraining-time decision or nothing.

---

## Summary

| Scheme | Fixes alignment? | Token cost | Position-tracking load | Retrofittable? | Sense of value? |
|---|---|---|---|---|---|
| L2R fixed chunks | No | Low | Low | — | No |
| R2L fixed chunks | Mostly | Low | Low | Unproven | No |
| Single-digit | Fully | High | High | No | No |

---

## The gap none of them touch

No tokenization choice here gives the model any signal that one number is numerically close to another. Magnitude only ever lives in arbitrary, unrelated token identities. That's a different layer of the problem, and it's what value-aware embeddings are aimed at.