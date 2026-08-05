from torch.utils.data import dataloader

from Attention.baseline import (
    SSelf_Attention,
    BatchSSelfAttention,
    SelfAttention,
    BatchSelfAttention
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


def test_selfAttention():
    attention = SelfAttention("The forest is filled with dark and evil creatures", query_input=4)
    tokenIDs = attention.tokenize()
    embeddings = attention.embed_tokens(tokenIDs)
    query, key, values = attention.QKV_init(embeddings)
    print(f"Query vector: {query}, Key Vector: {key}, Value Vector: {values}")
    weights = attention.attention_weight(key, query)
    context_vector = attention.context_vector(weights,values)
    print(f"Context vector for batch procedure: {context_vector}")

test_SSelf_Attention()
print("=" * 80)
test_BatchSSelfAttention()
print("=" * 80)
test_selfAttention()
print("=" * 80)

