import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.metricUtils import euclidean_metric
from utils.fewshotUtils import create_nshot_task_label

class PN(nn.Module):
    def __init__(self,baseModel,
                lstm_layers=1,
                lstm_input_size=1600,
                unrolling_steps=2,
                train_way=20,test_way=5,
                shot=5,query=5,query_val=15,
                ):
        super(PN,self).__init__()
        self.train_way = train_way
        self.test_way = test_way
        self.shot = shot
        self.query = query
        self.query_val = query_val

        self.baseModel = baseModel

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
        embeddings = self.baseModel(x)

        # Samples are ordered by the NShotWrapper class as follows:
        # k lots of n support samples from a particular class
        # k lots of q query samples from those classes
        support = embeddings[:self.shot * way]
        queries = embeddings[self.shot * way:]
        prototypes = compute_prototypes(support, way, self.shot)
        # Calculate squared distances between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
        distances = euclidean_metric(queries,prototypes)

        # Calculate log p_{phi} (y = k | x)
        y_pred = (distances).log_softmax(dim=1)
        label = create_nshot_task_label(way,query).cuda()
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
    class_prototypes = support.reshape(n, k, -1).mean(dim=0)
    return class_prototypes