from torch import nn
import torch

class Deep(nn.Module):
    def __init__(self, input_size, output_size, N):
        super(Deep, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, input_size))
        for _ in range(N - 1):
            self.layers.append(nn.Linear(input_size, input_size))
        self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x