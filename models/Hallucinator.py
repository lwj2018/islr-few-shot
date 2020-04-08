import torch.nn as nn
import torch
import torch.nn.functional as F

class Hallucinator(nn.Module):
    
    def __init__(self,f_dim):
        super(Hallucinator,self).__init__()
        self.fc1 = nn.Linear(f_dim,f_dim)
        self.fc2 = nn.Linear(f_dim,f_dim)
        self.fc3 = nn.Linear(f_dim,f_dim)

    def forward(self,x,z):
        x = x + z
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        return x