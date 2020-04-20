import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.metricUtils import euclidean_metric
from utils.fewshotUtils import create_nshot_task_label

class RN(nn.Module):
    def __init__(self,baseModel,
                lstm_layers=1,
                lstm_input_size=1600,
                unrolling_steps=2,
                train_way=20,test_way=5,
                shot=5,query=5,query_val=15,
                ):
        super(RN,self).__init__()
        self.train_way = train_way
        self.test_way = test_way
        self.shot = shot
        self.query = query
        self.query_val = query_val

        self.baseModel = baseModel
        self.relation = Relation()

    def forward(self, support, queries, mode='train'):
        if mode=='train':
            way = self.train_way
            query = self.query
        else:
            way = self.test_way
            query = self.query_val
        # Concatenate
        x = torch.cat([support,queries],0)
        # Embed all samples
        embeddings = self.baseModel.get_feature(x)

        # Samples are ordered by the NShotWrapper class as follows:
        # k lots of n support samples from a particular class
        # k lots of q query samples from those classes
        support = embeddings[:self.shot * way]
        queries = embeddings[self.shot * way:]
        prototypes = compute_prototypes(support, way, self.shot)
        # Calculate squared distances between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
        distances = self.relation(queries,prototypes)

        # Calculate log p_{phi} (y = k | x)
        y_pred = distances
        y_onehot = torch.zeros(y_pred.size()).cuda()
        label = create_nshot_task_label(way,query).unsqueeze(-1).cuda()
        y_onehot = y_onehot.scatter(1,label,1)
        y_onehot = y_onehot.float()
        label = label.squeeze(-1)
        return y_pred, label

    def get_optim_policies(self, lr):
        return [
            {'params':self.parameters(),'lr':lr},
        ]

def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.
    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task
    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape( (n, k,) + support.size()[-3:] ).mean(dim=0)
    return class_prototypes

class conv_block(nn.Module):
    def __init__(self,in_dim,h_dim=64):
        super(conv_block,self).__init__()
        self.conv = nn.Conv2d(in_dim,h_dim,3,padding=1)
        self.bn = nn.BatchNorm2d(h_dim)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class Relation(nn.Module):
    def __init__(self,in_dim=2*256,h_dim=64,z_dim=64):
        super(Relation,self).__init__()
        self.conv1 = conv_block(in_dim)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = conv_block(h_dim)
        self.fc1 = nn.Linear(z_dim,8)
        self.fc2 = nn.Linear(8,1)

    def forward(self,x1,x2):
        # Expand
        n = x1.shape[0]
        m = x2.shape[0]
        x1 = x1.unsqueeze(1).expand(n,m,-1,-1,-1)
        x2 = x2.unsqueeze(0).expand(n,m,-1,-1,-1)
        x = torch.cat([x1,x2],2)
        x = x.view( (-1,) + x.size()[-3:] )
        # Conv
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        # Flatten
        x = x.view(x.size(0),-1)
        # Fc
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        # Reshape
        x = x.view(n,m)
        return x