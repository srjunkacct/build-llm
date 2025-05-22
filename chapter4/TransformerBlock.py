import torch
import torch.nn as nn
from chapter3 import MultiHeadAttention
from chapter4 import FeedForward, LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention.MultiHeadAttention( d_in=cfg["emb_dim"],
                                       d_out=cfg["emb_dim"],
                                       context_length=cfg["context_length"],
                                       num_heads=cfg["n_heads"],
                                       dropout=cfg["drop_rate"],
                                       qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward.FeedForward(cfg)
        self.norm1 = LayerNorm.LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

