
import torch
import numpy

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(target.size()) == 2:
        target = target.argmax(-1)
    maxk = max(topk)
    batch_size = target.size(0)

    # return the k largest elements of the given input Tensor
    # along the given dimension. dim = 1
    # pred is the indices
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits