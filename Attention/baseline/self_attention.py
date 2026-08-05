
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
        key_i = x_i @ W_key
        value_i = x_i @ W_value

        return (query_i, key_i, value_i)

    def attention_weights(self, key, query):
        attention_scores = torch.dot(key, query)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return attention_weights

    def context_vector(self, attention_weights, value):
        context_vector = attention_weights @ value
        return context_vector



import torch

from Attention.baseline import SSelf_Attention

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