import torch
import torch.nn as nn
from GPT.baseline.dummyGPT import LayerNorm, FeedForward, ExampleDeepNeuralNetwork
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

embed_dim = 100
torch.manual_seed(123)
batch_example = torch.randn(2, embed_dim)
layer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
out = layer(batch_example)
print(out)
print("="*80)

# mean = out.mean(-1, keepdim=True)
# variance = out.var(-1, keepdim=True)
# print("Mean:", mean)
# print("Variance:", variance)

print("="*80)

ln = LayerNorm(emb_dim=embed_dim)
normalized_output = ln(batch_example)
mean = normalized_output.mean(-1, keepdim=True)
variance = normalized_output.var(-1, unbiased=False, keepdim=True)
# print("Mean:", mean)
# print("Variance:", variance)

print("="*80)

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.randn(2, 3, 768)
out = ffn(x)
print(out.shape)

print("="*80)

def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

layer_sizes = [3, 3, 3, 3, 3, 1]

sample_input = torch.tensor([[1., 0., -1.]])

torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
print_gradients(model_without_shortcut, sample_input)