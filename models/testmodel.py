from torch.nn import functional as F
from torch import nn
import torch
from torchsummary import summary

class TestConvModel(nn.Module):
    def __init__(self):
        super(TestConvModel, self).__init__()
        
        # spectrum convolutional input
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc01 = nn.Linear(16 * (343 // 4), 128)
        self.fc02 = nn.Linear(128, 64)
        
        # context input
        self.fc11 = nn.Linear(2, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc12 = nn.Linear(16, 32)

        # combine outputs
        self.fc = nn.Linear(64+32, 64)
        self.output = nn.Linear(64, 1)


    def forward(self, x, y):

        # forward pass for spectrum
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc01(x))
        x = F.leaky_relu(self.fc02(x))
        
        # forward pass for context
        y = self.bn1(F.leaky_relu(self.fc11(y)))
        y = F.leaky_relu(self.fc12(y))

        out = torch.cat((x, y), 1)
        out = F.leaky_relu(self.fc(out))
        out = self.output(out)
        
        return out

#if __name__ == "__main__":
    #a = torch.rand(32, 1, 343)
    #b = torch.rand(32, 2)
    #model = TestConvModel()
    #out = model(a, b)

    #summary(model)