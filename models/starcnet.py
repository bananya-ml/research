from torch import nn
import torch


class StarcNet(nn.Module):
    '''
    StarcNet network constructed from "STARCNET: Machine Learning for Star Cluster Identification"
    '''
    def __init__(self, input_dim = 5, size = 32, n=8, spectrum_width = 343, mag_dim = 10):
        super(StarcNet, self).__init__()
        self.a1 = 1 # filters multiplier
        self.a21 = 1
        self.a2 = 1
        self.a3 = 1
        n = 8 # for groupnorm
        self.sz1 = spectrum_width # size of input image (sz1 x sz1)
        self.sz = mag_dim # for secon input size (2*sz x 2*sz) default: 10
        
        self.net = nn.Sequential(ConvBlock(input_dim, 128, n),
                                 ConvBlock(128, 128, n),
                                 ConvBlock(128, 128, n),
                                 ConvBlock(128, 128, n),
                                 nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                 ConvBlock(128, 128, n),
                                 ConvBlock(128, 128, n),
                                 ConvBlock(128, 128, n))

        self.outblock = OutBlock(size, n)
        self.fc = nn.Linear(384, 4)
        
        self.resize = nn.Upsample(size=(self.sz1,self.sz1))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.GroupNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        '''
        x0 = x[:,:,:,:]#.unsqueeze(1)
        x1 = x[:,:,int(self.sz1/2-self.sz):int(self.sz1/2+self.sz), int(self.sz1/2-self.sz):int(self.sz1/2+self.sz)] #.unsqueeze(1)
        x2 = x[:,:,int(self.sz1/2-self.sz/2):int(self.sz1/2+self.sz/2+1), int(self.sz1/2-self.sz/2):int(self.sz1/2+self.sz/2+1)]#.unsqueeze(1)
        x1 = self.resize(x1)
        x2 = self.resize(x2)
        '''
        x0 = x[:]
        
        # first conv net
        out0 = self.net(x0)
        print(out0.shape)
        '''
        out0 = self.outblock(out0.view(-1, 128*int(self.sz1/2)*int(self.sz1/2)))
        
        # second conv net
        out1 = self.net(x1)
        out1 = self.outblock(out1.view(-1, 128*int(self.sz1/2)*int(self.sz1/2)))
         
        # third conv net
        out2 = self.net(x2)
        out2 = self.outblock(out2.view(-1, 128*int(self.sz1/2)*int(self.sz1/2)))
        
        # combine all 3 outputs to make a single prediction
        out = torch.cat((out0,out1,out2),1)
        out = self.fc(out)
        return out
        '''
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(n, out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))
    
class OutBlock(nn.Module):
    def __init__(self, size, n):
        super().__init__()
        self.fc = nn.Linear(128*int(size/2)*int(size/2), 128)
        self.gn = nn.GroupNorm(n,128)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        return self.dropout(self.relu(self.gn(self.fc(x))))
    
model = StarcNet(input_dim = 1)
a = torch.rand(1,343)
out = model(a.unsqueeze(1))