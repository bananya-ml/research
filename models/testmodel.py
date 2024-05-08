from torch.nn import functional as F
import torch
import math
from torch import nn
from torchsummary import summary

class TestConvModel(nn.Module):
    def __init__(self, input_channels=343, num_labels=1, hidden_dims=[256,512], filter_length=[3,5], num_filters=[4,16],
                 pool_length = 4, n = 1, mag = 2):
        super(TestConvModel, self).__init__()

        self.mag = mag
        self.mag_dim = input_channels/mag
        
        self.net = nn.Sequential(ConvBlock(1, num_filters[0], filter_length[0], pool_length, n),
                                ConvBlock(num_filters[0], num_filters[1], filter_length[1], pool_length, n))
        
        output_shape = self._compute_out_size((1, math.floor(input_channels/mag)), self.net)

        self.fc1 = nn.Linear(output_shape[0]*output_shape[1], hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.output = nn.Linear(2*hidden_dims[1], num_labels)
    
    def forward(self,x):
        
        x0 = x[:,:,:math.floor(self.mag_dim)]
        x1 = x[:,:,math.floor(self.mag_dim):2*math.floor(self.mag_dim)]
        
        x0 = self.fc2(self.fc1(torch.flatten(self.net(x0), 1)))
        
        x1 = self.fc2(self.fc1(torch.flatten(self.net(x1), 1)))
        
        out = torch.cat((x0, x1), 1)
        out = F.relu(self.output(out))
        
        return out
    
    def _compute_out_size(self,in_size, mod):
        """
        Compute output size of Module `mod` given an input with size `in_size`.
        """
        f = mod.forward(torch.autograd.Variable(torch.Tensor(1, *in_size)))
        return f.size()[1:]

class ConvBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim, filter_length, pool_length, n):
        super().__init__()
    
        self.conv = nn.Conv1d(input_dim, output_dim, filter_length)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(pool_length, pool_length)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


if __name__ == "__main__":
    a = torch.rand(32, 1, 343)
    model = TestConvModel()
    out = model(a)
    #summary(model, (1, 343))