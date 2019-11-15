#
# Script which contains the function necessary load the model 
#

import torch
from torch.nn import functional as F

class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (128, 128, 128, 2)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels = 2, out_channels = 2, kernel_size = (3,3,3), stride = 10, padding = 1)
        self.Flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(2704, 200)
        self.fc2 = torch.nn.Linear(200, 6)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.Flatten(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 1)

        return (x)
