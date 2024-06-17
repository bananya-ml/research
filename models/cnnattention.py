from torch.nn import functional as F
from torch import nn
import torch

class CNNWithAttention(nn.Module):
    def __init__(self):
        super(CNNWithAttention, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(4)
        self.mha = nn.MultiheadAttention(4, num_heads=2)
        self.scale = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv1d(4, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.fc01 = nn.Linear(64 * (343//4), 128) 
        self.fc02 = nn.Linear(128, 64)
        
        self.fc11 = nn.Linear(2, 64)
        self.fc12 = nn.Linear(64, 32)
        
        self.fc3 = nn.Linear(64+32, 1)

    def use_attention(self, x):
        
        bs, c, l = x.shape
        x_att = x.transpose(1, 2)
        
        x_att = self.norm(x_att)
        
        att_out, att_map = self.mha(x_att, x_att, x_att)
        
        return att_out.transpose(1, 2), att_map  # BSxCxL
    
    def forward(self, x, y):
        x = self.conv1(x)
        
        x = self.scale * self.use_attention(x)[0] + x
        #x, _ = self.mha(x, x, x)
        x = F.leaky_relu(x)
        x = self.maxpool(F.leaky_relu(self.conv2(x)))
        x = self.maxpool(F.leaky_relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc01(x))
        x = F.leaky_relu(self.fc02(x))
        
        y = F.leaky_relu(self.fc11(y))
        y  = F.leaky_relu(self.fc12(y))
        
        out = torch.cat((x,y), 1)
        out = F.leaky_relu(self.fc3(out))
    
        return out