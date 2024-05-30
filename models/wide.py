from torch import nn
import torch

class Wide(nn.Module):
    def __init__(self, input_size, output_size, N):
        super(Wide, self).__init__()
        self.fc1 = nn.Linear(input_size, N)
        self.fc2 = nn.Linear(N, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x