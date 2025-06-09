from torch.utils.data import DataLoader
from chapter6.SpamDataSet import SpamDataSet
import torch
import tiktoken
num_workers = 0
batch_size = 8
torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataSet( csv_file="train.csv", max_length=None, tokenizer=tokenizer)
test_dataset = SpamDataSet(csv_file="test.csv", max_length=None, tokenizer=tokenizer)
val_dataset = SpamDataSet( csv_file="validation.csv", max_length=None, tokenizer=tokenizer)

train_loader = DataLoader( dataset=train_dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           drop_last=True)

test_loader = DataLoader( dataset=test_dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           drop_last=True)

val_loader = DataLoader( dataset=val_dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           drop_last=True)

for input_batch, target_batch in train_loader:
    pass

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions:", target_batch.shape)