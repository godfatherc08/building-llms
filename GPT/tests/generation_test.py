import torch

from Tokenizers.baseline.tokenizers import SimpleTokenizerV2, build_vocab
from GPT.baseline.GPT import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 33,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

model = GPTModel(GPT_CONFIG_124M)
start_context = "Hello"
SAMPLE_CORPUS = (
    "Hello, world. This is a small test corpus. "
    "It has commas, periods; and colons: plus a dash -- here. "
    "The quick brown fox jumps over the lazy dog?"
)
vocab = build_vocab(SAMPLE_CORPUS, add_special_tokens=True)
tokenizer = SimpleTokenizerV2(vocab)

encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=1,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))

text = out.squeeze(0).tolist()
decoded_text = tokenizer.decode(text)
print(decoded_text)
