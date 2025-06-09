import torch
import tiktoken
from chapter4.GenerateTextUtils import generate_text_simple
from chapter4.GPTModel import GPTModel

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


GPT_CONFIG_124M={ "vocab_size": 50257,
                  "context_length": 256,
                  "emb_dim": 768,
                  "n_heads": 12,
                  "n_layers": 12,
                  "drop_rate": 0.1,
                  "qkv_bias": False }

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}


