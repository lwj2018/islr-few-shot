import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.metricUtils import euclidean_metric
from utils.critUtils import convert_to_onehot

class GCR_ri(nn.Module):
    def __init__(self,baseModel,genModel,global_base=None,global_novel=None,
                train_way=20,test_way=5,
                shot=5,query=5,query_val=15,f_dim=1600,
                z_dim=512):
        super(GCR_ri,self).__init__()
        self.train_way = train_way
        self.test_way = test_way
        self.shot = shot
        self.query = query
        self.query_val = query_val
        self.f_dim = f_dim
        self.z_dim = z_dim

        self.baseModel = baseModel
        self.genModel = genModel
        self.registrator = Registrator(f_dim,z_dim)
        self.relation1 = Relation1(2*f_dim)
        self.relation2 = Relation2(2*z_dim)
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

        if mode == 'train':
            which_novel = torch.gt(gt,79)
            which_base = way-torch.numel(gt[which_novel])

            if which_base < way:
                proto_base = proto[:,:which_base,:]
                proto_novel = proto[:,which_base:,:]
                # Synthesis module corresponds to section 3.2 of the thesis
                noise = torch.cuda.FloatTensor((way-which_base)*self.shot, self.f_dim).normal_()
                proto_novel_gen = self.genModel(proto_novel.reshape(self.shot*(way-which_base),-1), noise)
                proto_novel_gen = proto_novel_gen.reshape(self.shot, way-which_base, -1)
                proto_novel_wgen = torch.cat([proto_novel,proto_novel_gen])
                ind_gen = torch.randperm(2*self.shot)
                proto_novel_f = proto_novel_wgen[ind_gen[:self.shot],:,:]
                # Corresponds to episodic repesentations in the thesis
                proto = torch.cat([proto_base, proto_novel_f],1)

        proto_final = self.induction(proto)

        # shape of global_new is: total_class(100) x z_dim(512)
        # shape of proto_new is: way(20 or 5) x z_dim(512)
        global_new, proto_new = self.registrator(support_set=torch.cat([self.global_base,self.global_novel]), query_set=proto_final)
        # shape of the dist_metric is: way x total_class
        logits2 = self.relation2(proto_new, global_new)

        similarity = F.normalize(logits2,1,-1)
        # similarity = logits2
        feature = torch.matmul(similarity, torch.cat([self.global_base,self.global_novel]))
        # shape of data_query is: (query x way) x ...
        # shape of feature is: way x f_dim(1600)
        # so the shape of result is (query x way) x way
        logits = self.relation1(self.baseModel(data_query),feature)
        label = torch.arange(way).repeat(query)
        label = label.type(torch.cuda.LongTensor)

        gt3 = gt.repeat(self.shot)
        logits3 = self.relation1(proto.reshape(self.shot*way,-1),torch.cat([self.global_base,self.global_novel]))

        return logits, label, logits2, gt, logits3, gt3

    def get_optim_policies(self,lr):
        return [
            {'params':self.genModel.parameters(),'lr':lr},
            {'params':self.registrator.parameters(),'lr':lr},
            {'params':self.relation1.parameters(),'lr':lr},
            {'params':self.relation2.parameters(),'lr':lr},
            {'params':self.global_base,'lr':lr},
            {'params':self.global_novel,'lr':lr},
            {'params':self.induction.parameters(),'lr':lr}
        ]

    def get_finetune_policies(self,lr):
        return [
            {'params':self.genModel.parameters(),'lr':lr},
            {'params':self.registrator.parameters(),'lr':lr},
            {'params':self.relation1.parameters(),'lr':lr},
            {'params':self.relation2.parameters(),'lr':lr},
            {'params':self.global_base,'lr':lr},
            {'params':self.global_novel,'lr':lr},
            {'params':self.induction.parameters(),'lr':lr*10}
        ]
        

class Registrator(nn.Module):
    def __init__(self,f_dim,z_dim):
        super(Registrator, self).__init__()
        self.fc_params_support = nn.Sequential(
        	torch.nn.Linear(1600, 512),
        	torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
        	)
        self.fc_params_query = nn.Sequential(
        	torch.nn.Linear(1600, 512),
        	torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
        	)

    def forward(self, support_set, query_set):
        support_set_2 = self.fc_params_support(support_set)
        query_set_2 = self.fc_params_query(query_set)
        return support_set_2, query_set_2

class Relation1(nn.Module):
    def __init__(self,h_dim):
        super(Relation1, self).__init__()
        self.fc1 = nn.Linear(h_dim,1600)
        self.fc2 = nn.Linear(1600,200)
        self.fc3 = nn.Linear(200,1)

    def forward(self,x1,x2):
        n = x1.shape[0]
        m = x2.shape[0]
        x1 = x1.unsqueeze(1).expand(n, m, -1)
        x2 = x2.unsqueeze(0).expand(n, m, -1)
        x = torch.cat([x1,x2],2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = x.squeeze(-1)
        return x

class Relation2(nn.Module):
    def __init__(self,h_dim):
        super(Relation2, self).__init__()
        self.fc1 = nn.Linear(h_dim,512)
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64,1)

    def forward(self,x1,x2):
        n = x1.shape[0]
        m = x2.shape[0]
        x1 = x1.unsqueeze(1).expand(n, m, -1)
        x2 = x2.unsqueeze(0).expand(n, m, -1)
        x = torch.cat([x1,x2],2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = x.squeeze(-1)
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