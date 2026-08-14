
import torch

from Attention.baseline import SSelf_Attention

class SelfAttention(SSelf_Attention):
    def __init__(self, input_string:str, query_input:int):
        super().__init__(input_string, query_input)

    def QKV_init(self, input_embeddings):
        x_i = input_embeddings[self.query_input]
        input_embeddings_shape = x_i.shape[0]
        output_embeddings_shape = input_embeddings_shape #keep the gpt config

        torch.manual_seed(123)
        W_query = torch.nn.Parameter(torch.rand(input_embeddings_shape, output_embeddings_shape), requires_grad=False)
        W_key = torch.nn.Parameter(torch.rand(input_embeddings_shape, output_embeddings_shape), requires_grad=False)
        W_value= torch.nn.Parameter(torch.rand(input_embeddings_shape, output_embeddings_shape), requires_grad=False)

        # requiregrad will be true during training to allow tuning of weights

        query_i = x_i @ W_query
        key_i = input_embeddings @ W_key
        value_i = input_embeddings @ W_value

        return (query_i, key_i, value_i)

    def attention_score(self, key, query):
        print(f"query: {query}")
        print(f"key : {key}")
        attention_scores = query @ key.T
        print(f"Attention score: {attention_scores}")
        return attention_scores

    def attention_weight(self, attention_scores, key):
        d_k = key.shape[-1]
        attention_weights = torch.softmax(attention_scores / d_k ** 0.5, dim=-1)
        return attention_weights

    def context_vector(self, attention_weights, value):
        print(f"Attention weights: {attention_weights}")
        print(f"value: {value}")
        context_vector = attention_weights @ value
        return context_vector



from Attention.baseline import SSelf_Attention
# not complete
class BatchSelfAttention(SSelf_Attention):
    def __init__(self, input_string:str):
        super().__init__(input_string, query_input=0)
        self.query_input = None

    def attention_score(self, input_embeddings):
        attention_scores = input_embeddings @ input_embeddings.T
        return attention_scores

    def context_vector(self, attention_weights, input_embeddings):
        context_vector = attention_weights @ input_embeddings
        return context_vector

class CasualAttention(SelfAttention):
    def __init__(self, input_string: str, query_input: int):
        super().__init__(input_string, query_input)

    def attention_score_mask(self, attention_score, mask):
        attention_mask = attention_score.masked_fill(mask.bool(), -float('inf'))
        print(f"Attention score: {attention_score}")
        return attention_mask

    def masking(self, key, attention_score, attention_weights):
        context_length = attention_score.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked = self.attention_score_mask(attention_score, mask)

        mask_simple = torch.tril(torch.ones(context_length, context_length))
        print(f"Mask simple: {mask_simple}")

        masked_simple = attention_weights*mask_simple
        print(f"Masked simple: {masked_simple}")

        row_sums = masked_simple.sum(dim=1, keepdim=True)
        masked_simple_normalized = masked_simple/row_sums
        print(f"Masked simple normalized: {masked_simple_normalized}")

        mask_attn_weights = torch.softmax(masked/key.shape[-1]**0.5, dim=-1)
        print(f"Masked attention weights: {mask_attn_weights}")

