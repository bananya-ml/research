import torch
import torch.nn as nn
from torch.nn import functional as F

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(2,64)
        self.fc2 = nn.Linear(64,32)

        self.fc = nn.Linear((128*343)+32, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x, y):
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(x.size(0), -1)
            
        y = F.leaky_relu(self.fc1(y))
        y = F.leaky_relu(self.fc2(y))

        out = torch.cat((x,y), 1)
        out = F.leaky_relu(self.fc(out))
        out = self.output(out)
        return out