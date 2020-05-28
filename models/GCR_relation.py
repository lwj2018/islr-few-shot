import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.metricUtils import euclidean_metric
from utils.critUtils import convert_to_onehot

class GCR_relation(nn.Module):
    def __init__(self,baseModel,global_base=None,global_novel=None,
                train_way=20,test_way=5,
                shot=5,query=5,query_val=15,f_dim=1024,
                z_dim=512):
        super(GCR_relation,self).__init__()
        self.train_way = train_way
        self.test_way = test_way
        self.shot = shot
        self.query = query
        self.query_val = query_val
        self.f_dim = f_dim
        self.z_dim = z_dim

        self.baseModel = baseModel
        self.registrator = Registrator(f_dim,z_dim)
        self.relation1 = Relation1(2*f_dim)
        self.relation2 = Relation2(2*z_dim)
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

        if mode=='train':
            which_novel = torch.gt(gt,79)
            which_base = way-torch.numel(gt[which_novel])
            if which_base < way:
                proto_base = proto[:,:which_base,:]
                proto_novel = proto[:,which_base:,:]
                proto_base = proto_base.mean(dim=0)
                if self.shot>1:
                    # Synthesis module corresponds to section 3.2 of the thesis
                    # Temporarily do not use Hallucinator
                    ind_gen = torch.randperm(self.shot)
                    train_num = np.random.randint(1,self.shot)
                    proto_novel_f = proto_novel[ind_gen[:train_num],:,:]
                    weight_arr = np.random.rand(train_num)
                    weight_arr = weight_arr/np.sum(weight_arr)
                    # Generate a new sample
                    # shape of proto_novel_f is: shot x novel_class x f_dim(1600)
                    proto_novel_f = (torch.from_numpy(weight_arr.reshape(-1,1,1)).type(torch.float).cuda()*proto_novel_f).sum(dim=0)
                    # Corresponds to episodic repesentations in the thesis
                    # After sum or mean, shape of protos are: class x f_dim(1600)
                else:
                    proto_novel_f = proto_novel.mean(dim=0)
                proto_final = torch.cat([proto_base, proto_novel_f],0)
            else:
                proto_final = proto.reshape(self.shot,way,-1).mean(dim=0)
        else:
            proto_final = proto.mean(dim=0)

        # shape of global_new is: total_class(100) x z_dim(512)
        # shape of proto_new is: way(20 or 5) x z_dim(512)
        global_new, proto_new = self.registrator(support_set=torch.cat([self.global_base,self.global_novel]), query_set=proto_final)
        # shape of the dist_metric is: way x total_class
        logits2 = self.relation2(proto_new, global_new)

        # similarity = F.normalize(logits2,1,-1)
        similarity = logits2
        feature = torch.matmul(similarity, torch.cat([self.global_base,self.global_novel]))
        # shape of data_query is: (query x way) x ...
        # shape of feature is: way x f_dim(1600)
        # so the shape of result is (query x way) x way
        q_proto = self.baseModel(data_query)
        logits = self.relation1(q_proto,feature)
        label = torch.arange(way).repeat(query)
        label = label.type(torch.cuda.LongTensor)

        gt3 = gt.repeat(query)
        logits3 = self.relation1(q_proto.reshape(query*way,-1),torch.cat([self.global_base,self.global_novel]))

        return logits, label, logits2, gt, logits3, gt3

    def get_optim_policies(self,lr):
        return [
            {'params':self.registrator.parameters(),'lr':lr},
            {'params':self.relation1.parameters(),'lr':lr},
            {'params':self.relation2.parameters(),'lr':lr},
            {'params':self.global_base,'lr':lr},
            {'params':self.global_novel,'lr':lr}
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
        x = self.fc3(x)
        # x = F.sigmoid(x)
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