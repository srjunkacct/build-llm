import torch

from chapter2.DataLoaderV1 import create_dataloader_v1
from chapter5.TokenUtilities import GPT_CONFIG_124M
file_path = "/Users/edwardlee/git/build-llm/chapter2/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

torch.manual_seed(123)
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
train_loader = create_dataloader_v1(train_data,
                                    batch_size=2,
                                    max_length=GPT_CONFIG_124M["context_length"],
                                    stride=GPT_CONFIG_124M["context_length"],
                                    drop_last=True,
                                    shuffle=True,
                                    num_workers=0)
val_loader = create_dataloader_v1(val_data,
                                  batch_size=2,
                                  max_length=GPT_CONFIG_124M["context_length"],
                                  stride=GPT_CONFIG_124M["context_length"],
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=0)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x,y in val_loader:
    print(x.shape, y.shape)





