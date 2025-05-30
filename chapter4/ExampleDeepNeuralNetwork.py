import torch
import torch.nn as nn

from chapter4 import GELU

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1], GELU.GELU())),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2], GELU.GELU())),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3], GELU.GELU())),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4], GELU.GELU())),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5], GELU.GELU()))
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x =  x + layer_output
            else:
                x = layer_output
        return x