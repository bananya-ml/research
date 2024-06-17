import torch
import torch.nn as nn
from torch.nn import functional as F

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)

        self.fc = nn.Linear(128+32, 128)
        self.output = nn.Linear(128, 1)



    def forward(self, x, y):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        x = h[-1]

        y = F.leaky_relu(self.fc1(y))
        y = F.leaky_relu(self.fc2(y))
        
        out = torch.cat((x,y),1)
        out = F.leaky_relu(self.fc(out))
        out = self.output(out)
        return out