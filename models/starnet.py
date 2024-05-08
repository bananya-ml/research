import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class StarNet(nn.Module):
    def __init__(self, num_fluxes, num_filters, filter_length, 
                 pool_length, num_hidden, num_labels):
        super().__init__()
        
        self.num_labels = num_labels
        # Convolutional and pooling layers
        self.conv1 = nn.Conv1d(1, num_filters[0], filter_length)
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], filter_length)
        self.pool = nn.MaxPool1d(pool_length, pool_length)
        
        # Determine shape after pooling
        pool_output_shape = self.compute_out_size((1,num_fluxes), 
                                             nn.Sequential(self.conv1, 
                                                           self.conv2, 
                                                           self.pool))
        
        # Fully connected layers
        self.fc1 = nn.Linear(pool_output_shape[0]*pool_output_shape[1], num_hidden[0])
        self.fc2 = nn.Linear(num_hidden[0], num_hidden[1])
        self.output = nn.Linear(num_hidden[1], num_labels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.num_labels == 1:
            x = F.sigmoid(self.output(x))
        else:
            x = self.output(x)
        return x
    
    def compute_out_size(self, in_size, mod):
        """
        Compute output size of Module `mod` given an input with size `in_size`.
        """
        
        f = mod.forward(autograd.Variable(torch.Tensor(1, *in_size)))
        return f.size()[1:]