from torch.nn.functional import softmax

from Tokenizers.baseline import get_bpe_tokenizer, create_token_embedding_layer, create_positional_embedding_layer, compute_input_embeddings, create_dataloader_v1
import torch

class SSelf_Attention():
    def __init__(self, input_string:str, query_input:int):
        self.query_input = query_input
        self.input_string = input_string

    def tokenize(self):
        tokenizer = get_bpe_tokenizer('gpt2')
        tokenIDs = tokenizer.encode(self.input_string)
        return tokenIDs

    def embed_tokens(self, tokenIDs):
        GPT2_VOCAB_SIZE = 50257
        token_ids = torch.tensor(tokenIDs).unsqueeze(0)
        print(token_ids.shape)
        print(token_ids)
        seq_len = token_ids.shape[1]
        output_dim = 256
        token_embedding_layer = create_token_embedding_layer(GPT2_VOCAB_SIZE, output_dim)
        positional_embedding_layer = create_positional_embedding_layer(seq_len, output_dim)
        input_embeddings = compute_input_embeddings(token_ids, token_embedding_layer, positional_embedding_layer)
        return input_embeddings.squeeze(0) # served its purpose for passing the 2d input required for compute_input_embeddings

    def attention_score(self, input_embeddings):
        query = input_embeddings[self.query_input]
        shape = input_embeddings.shape[0]
        attention_scores = torch.empty(shape)
        for i, x_i in enumerate(input_embeddings):
            attention_scores[i] = torch.dot(x_i, query)
        return attention_scores

    def softmax(x):
        return torch.softmax(x, dim=-1)

    def attention_weight(self, attention_scores):
        attention_weights = softmax(attention_scores, dim=-1)
        return attention_weights

    def context_vector(self, attention_weights, input_embeddings):
        query = input_embeddings[self.query_input]
        context_vector = torch.zeros(query.shape[0])

        for i, x_i in enumerate(input_embeddings):
            context_vector[i] += torch.dot(x_i, query)
        return context_vector



from Attention.baseline import SSelf_Attention

class BatchSSelfAttention(SSelf_Attention):
    def __init__(self, input_string:str):
        super().__init__(input_string, query_input=0)
        self.query_input = None

    def attention_score(self, input_embeddings):
        attention_scores = input_embeddings @ input_embeddings.T
        return attention_scores

    def context_vector(self, attention_weights, input_embeddings):
        context_vector = attention_weights @ input_embeddings
        return context_vector
