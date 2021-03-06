'''
    Without relation module, simply expand the model capacity
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.metricUtils import euclidean_metric
from utils.critUtils import convert_to_onehot

class GCR_ex(nn.Module):
    def __init__(self,baseModel,global_base=None,global_novel=None,
                train_way=20,test_way=5,
                shot=5,query=5,query_val=15,f_dim=1024,
                z_dim=512):
        super(GCR_ex,self).__init__()
        self.train_way = train_way
        self.test_way = test_way
        self.shot = shot
        self.query = query
        self.query_val = query_val
        self.f_dim = f_dim
        self.z_dim = z_dim

        self.baseModel = baseModel
        self.registrator = Registrator(f_dim,z_dim)
        self.extra1 = Extra1(f_dim)
        self.extra2 = Extra2(z_dim)
        self.induction = Induction(f_dim)
        self.global_base = global_base
        self.global_novel = global_novel

    def forward(self, data_shot, data_query, lab, mode='train'):
        if mode=='train':
            way = self.train_way
            query = self.query
        else:
            way = self.test_way
            query = self.query_val

        p = self.shot * way
        gt = lab[:p].reshape(self.shot, way)[0,:]
        proto = self.baseModel(data_shot)
        proto = proto.reshape(self.shot, way, -1)

        proto_final = self.induction(proto)

        # shape of global_new is: total_class(100) x z_dim(512)
        # shape of proto_new is: way(20 or 5) x z_dim(512)
        global_new, proto_new = self.registrator(support_set=torch.cat([self.global_base,self.global_novel]), query_set=proto_final)
        # shape of the dist_metric is: way x total_class
        logits2 = euclidean_metric(self.extra2(proto_new), self.extra2(global_new))

        similarity = F.normalize(logits2,1,-1)
        # similarity = logits2
        feature = torch.matmul(similarity, torch.cat([self.global_base,self.global_novel]))
        # shape of data_query is: (query x way) x ...
        # shape of feature is: way x f_dim(1600)
        # so the shape of result is (query x way) x way
        q_proto = self.baseModel(data_query)
        logits = euclidean_metric(self.extra1(q_proto),self.extra1(feature))
        label = torch.arange(way).repeat(query)
        label = label.type(torch.cuda.LongTensor)

        gt3 = gt.repeat(query)
        # logits3 = euclidean_metric(proto.reshape(self.shot*way,-1),torch.cat([self.global_base,self.global_novel]))
        logits3 = euclidean_metric(self.extra1(q_proto.reshape(query*way,-1)),self.extra1(torch.cat([self.global_base,self.global_novel])))

        return logits, label, logits2, gt, logits3, gt3

    def get_feature(self, x):
        return self.baseModel(x)

    def get_optim_policies(self,lr):
        return [
            {'params':self.registrator.parameters(),'lr':lr},
            {'params':self.global_base,'lr':lr},
            {'params':self.global_novel,'lr':lr},
            {'params':self.induction.parameters(),'lr':lr}
        ]

    def get_finetune_policies(self,lr):
        return [
            {'params':self.registrator.parameters(),'lr':lr},
            {'params':self.global_base,'lr':lr},
            {'params':self.global_novel,'lr':lr},
            {'params':self.induction.parameters(),'lr':lr*10}
        ]
        

class Registrator(nn.Module):
    def __init__(self,f_dim,z_dim):
        super(Registrator, self).__init__()
        self.fc_params_support = nn.Sequential(
        	torch.nn.Linear(f_dim, z_dim),
        	torch.nn.BatchNorm1d(z_dim),
                torch.nn.ReLU(),
        	)
        self.fc_params_query = nn.Sequential(
        	torch.nn.Linear(f_dim, z_dim),
        	torch.nn.BatchNorm1d(z_dim),
                torch.nn.ReLU(),
        	)

    def forward(self, support_set, query_set):
        support_set_2 = self.fc_params_support(support_set)
        query_set_2 = self.fc_params_query(query_set)
        return support_set_2, query_set_2

class Extra1(nn.Module):
    def __init__(self,h_dim):
        super(Extra1, self).__init__()
        self.fc1 = nn.Linear(h_dim,1600)
        self.fc2 = nn.Linear(1600,200)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Extra2(nn.Module):
    def __init__(self,h_dim):
        super(Extra2, self).__init__()
        self.fc1 = nn.Linear(h_dim,512)
        self.fc2 = nn.Linear(512,64)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Induction(nn.Module):
    def __init__(self,f_dim):
        super(Induction, self).__init__()
        self.lstm = nn.LSTM(input_size=f_dim,hidden_size=f_dim,num_layers=2,
                bidirectional=True)
        self.fc = nn.Linear(2*f_dim,f_dim)

    def forward(self,x):
        # shape of x is: T(shot) x N(way) x f_dim
        h,_ = self.lstm(x)
        x = self.fc(h)
        # Average Pool
        x = torch.mean(x,dim=0)
        return x