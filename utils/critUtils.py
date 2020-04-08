import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_to_onehot(x,num_class):
    x = x.unsqueeze(-1)
    n = x.size(0)
    x_oneshot = torch.zeros(n,num_class)
    x_oneshot = x_oneshot.type(torch.cuda.FloatTensor)
    x_oneshot.scatter_(1,x,1)
    return x_oneshot

class loss_for_gcr(nn.Module):

    def __init__(self):
        super(loss_for_gcr,self).__init__()

    def forward(self, logits, label, logits2, train_gt):
        loss1 = F.cross_entropy(logits, label)
        loss2 = F.cross_entropy(logits2, train_gt)
        loss = loss1+loss2
        return loss, loss1, loss2

class loss_for_gcr_relation(nn.Module):

    def __init__(self):
        super(loss_for_gcr_relation,self).__init__()

    def forward(self, logits, label, logits2, train_gt, logits3, gt3):
        loss1 = F.cross_entropy(logits, label)
        loss2 = F.cross_entropy(logits2, train_gt)
        loss3 = F.cross_entropy(logits3, gt3)
        loss = loss1+loss2+loss3
        return loss, loss1, loss2, loss3