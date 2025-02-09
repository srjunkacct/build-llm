import torch
import torch.nn as nn

class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value =nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys =self.W_key(x)
        queries =  self.W_query(x)
        values = self.W_value(x)
        attention_scores = queries @ keys.T # omega
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vector = attention_weights @ values
        return context_vector