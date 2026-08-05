# Attention

## What it does

Attention lets each token pull information from the other tokens in the sequence instead of being processed in isolation. 
## What this reproduction currently has

self-attention with and without QKV — queries, keys, and values computed over the full sequence, with every token free to attend to every other token in both directions.

## What I want to improve

**Causal attention.** The current block lets a token see tokens that come after it. For a language model that's cheating — at generation time the future doesn't exist yet. I want to add a causal mask so each token only attends to itself and the positions before it. This is the version that matches how the model is actually used.

**Multi-head attention.** Right now there's a single attention computation. Splitting it into several heads lets the model attend to different kinds of relationships at once (e.g. one head tracking syntax, another tracking longer-range reference) and then combine them. I want a multi-head class that runs the heads in parallel and concatenates their outputs.

**`nn.Linear` vs manual weight matrices.** The current implementation builds the query/key/value projections as raw `nn.Parameter` matrices. I want to compare that against using `nn.Linear` layers, which is the more standard choice. Points to check: whether the initialization differs, whether a bias term changes results, and whether the two are numerically equivalent when bias is off. Worth confirming they line up before building the bigger classes on top, so the baseline stays trustworthy.