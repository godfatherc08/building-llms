from torch.utils.data import dataloader

from Attention.baseline import (
    SSelf_Attention,
    BatchSSelfAttention
)

def test_SSelf_Attention():
    attention = SSelf_Attention("The forest is filled with dark and evil creatures", 4)
    tokenIDs = attention.tokenize()
    print(tokenIDs)
    print("=" * 80)
    embeddings = attention.embed_tokens(tokenIDs)
    print(embeddings)
    print("=" * 80)
    scores = attention.attention_score(embeddings)
    print(scores)
    weights = attention.attention_weight(scores)
    print(weights)
    context_vector = attention.context_vector(weights, embeddings)
    print(f"Final Context Vector: {context_vector}")


def test_BatchSSelfAttention():
    attention = BatchSSelfAttention("The forest is filled with dark and evil creatures")
    tokenIDs = attention.tokenize()
    embeddings = attention.embed_tokens(tokenIDs)
    scores = attention.attention_score(embeddings)
    print(scores)
    weights = attention.attention_weight(scores)
    context_vector = attention.context_vector(weights,embeddings)
    print(f"Context vector for batch procedure: {context_vector}")


test_SSelf_Attention()
print("=" * 80)
test_BatchSSelfAttention()
