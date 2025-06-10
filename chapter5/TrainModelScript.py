import tiktoken
import torch

from chapter2.DataLoaderV1 import create_dataloader_v1
from chapter4.Config import GPT_CONFIG_124M
from chapter4.GPTModel import GPTModel
from chapter5.TrainModel import train_model_simple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10

file_path = "../chapter2/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

torch.manual_seed(123)
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
train_loader = create_dataloader_v1(train_data,
                                    batch_size=2,
                                    #max_length=GPT_CONFIG_124M["context_length"],
                                    #stride=GPT_CONFIG_124M["context_length"],
                                    drop_last=True,
                                    shuffle=True,
                                    num_workers=0)
val_loader = create_dataloader_v1(val_data,
                                  batch_size=2,
                                  #max_length=GPT_CONFIG_124M["context_length"],
                                  #stride=GPT_CONFIG_124M["context_length"],
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=0)

train_losses, val_losses, tokens_seen = train_model_simple(model,
                                                           train_loader,
                                                           val_loader,
                                                           optimizer,
                                                           device,
                                                           num_epochs=num_epochs,
                                                           eval_freq=5,
                                                           eval_iter=5,
                                                           start_context="Every effort moves you",
                                                           tokenizer=tokenizer)