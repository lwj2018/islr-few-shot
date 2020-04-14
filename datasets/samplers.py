import torch
import numpy as np
import time

class CategoriesSampler_train():
    def __init__(self, label, n_batch, n_cls, n_shot,n_query, n_base_class):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_base_class = n_base_class
        self.n_reserve = 40

        label_set = list(set(label))
        self.label_set = label_set
        label = np.array(label)
        self.m_ind = {}
        for i in label_set:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            start = time.time()
            batch = []
            query_batch = []
            shot_batch = []
            classes = np.random.permutation(self.label_set)[:self.n_cls]
            classes.sort()
            for c in classes:
                if c < self.n_base_class:
                    l = self.m_ind[c]
                    tmp = torch.randperm(self.n_reserve)
                    batch.append(l[tmp[:self.n_shot+self.n_query]])

                else:
                    # 如果c属于c_novel，只取前n_shot个样本用于训练，然后做数据增强
                    l = self.m_ind[c]
                    if self.n_shot>1:
                        tmp = torch.randperm(self.n_shot)
                        novel_query = torch.randperm(self.n_shot-1)[0]+1
                        a = tmp[:self.n_shot-novel_query]
                        b = tmp[self.n_shot-novel_query:]
                        batch.append(torch.cat((l[a.repeat(15)[:self.n_shot]],l[b.repeat(15)[:self.n_query]])))
                    else:
                        ind = l[0].view(-1)
                        batch.append(torch.cat([ind,ind]))

            # shape of batch is (n_shot+n_query) x train_way
            batch = torch.stack(batch).t().reshape(-1)
            end = time.time()
            yield batch

class CategoriesSampler_val():
    def __init__(self, label, n_batch, n_cls, n_shot,n_query):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query

        label_set = list(set(label))
        self.label_set = label_set
        label = np.array(label)
        self.m_ind = {}
        for i in label_set:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # 随机选出n_cls个类
            classes = np.random.permutation(self.label_set)[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                # 前n_shot个样本作为支撑集，之后的样本中取n_query个作为query set
                pos = torch.cat([torch.Tensor(range(0,self.n_shot)).type(torch.LongTensor),self.n_shot+torch.randperm(len(l)-self.n_shot)[:self.n_query]])
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class PretrainSampler():
    def __init__(self, label, n_shot, n_base_class, batch_size):
        self.n_shot = n_shot
        self.n_base_class = n_base_class
        self.batch_size = batch_size
        # number of samples in a class, !! Specific setting for isl
        n_samples = 50
        n_reserve = 40

        idx_list = []
        for idx, lb in enumerate(label):
            if lb < n_base_class:
                if idx % n_samples < n_reserve:
                    idx_list.append(idx)
            else:
                if idx % n_samples < n_shot:
                    idx_list.append(idx)
        idx_list = np.array(idx_list)
        idx_list = np.random.permutation(idx_list)
        self.idx_list = torch.Tensor(idx_list).type(torch.LongTensor)

        n = len(self.idx_list) // self.batch_size
        r = len(self.idx_list) % self.batch_size
        if r>0 : n += 1
        self.n_batch = n

    def __len__(self): 
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            if (i_batch+1)*self.batch_size < len(self.idx_list):
                batch = self.idx_list[i_batch*self.batch_size:(i_batch+1)*self.batch_size]
            else:
                batch = self.idx_list[i_batch*self.batch_size:]
            yield batch