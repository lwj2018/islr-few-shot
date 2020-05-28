import torch.nn as nn
import torch
import torch.nn.functional as F
from models.HCN import hcn
from models.Hallucinator import Hallucinator

class HCN_GEN(nn.Module):
    
    def __init__(self,out_dim, f_dim=1024):
        super(HCN_GEN,self).__init__()
        self.hcn = hcn(out_dim)
        self.gen = Hallucinator(f_dim)
        
    def forward(self, x):
        x = self.hcn.get_feature(x)
        z = torch.cuda.FloatTensor(x.size()).normal_()
        x = self.gen(x,z)
        x = self.hcn.classify(x)
        return x

    def get_feature(self, x):
        x = self.hcn.get_feature(x)
        return x