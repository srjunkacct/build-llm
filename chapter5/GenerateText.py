import torch
import tiktoken
from chapter4.GPTModel import GPTModel
from chapter5.LoadWeights import load_weights_into_gpt
from chapter5.TokenUtilities import text_to_token_ids, token_ids_to_text
from chapter5.TrainModel import generate
from gpt_download import download_and_load_gpt2

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)" : { "emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)" : { "emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)" : { "emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)" : { "emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])



model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.to(device)
torch.manual_seed(123)
token_ids = generate( model=model,
                      idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
                      max_new_tokens=26,
                      context_size=BASE_CONFIG["context_length"],
                      top_k=50,
                      temperature=1.5)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


