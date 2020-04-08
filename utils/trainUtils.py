import torch
import torch.nn.functional as F
import numpy
import time
from utils.metricUtils import *
from utils.Averager import AverageMeter
from utils.Recorder import Recorder
from utils.fewshotUtils import create_nshot_task_label
from collections import OrderedDict

def train_cnn(model, criterion, optimizer, trainloader, 
        device, epoch, log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    avg_proto = AverageMeter()
    # Create recorder
    averagers = [losses, avg_acc]
    names = ['train loss','train acc']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set trainning mode
    model.train()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(trainloader,1):
        # measure data loading time
        recoder.data_tok()

        # get the data and labels
        data,lab = [_.to(device) for _ in batch]

        p = args.shot * args.train_way
        data_shot = data[:p]
        data_query = data[p:]
        data_shot = data_shot[:,:3,:]
        data_query = data_query[:,:3,:]
        input = torch.cat([data_shot,data_query],0)

        optimizer.zero_grad()
        # forward
        outputs = model(input)

        # compute the loss
        loss = criterion(outputs,lab)

        # backward & optimize
        loss.backward()
        optimizer.step()

        # Calculate global proto
        proto = model.get_feature(input)
        episodic_proto = numpy.zeros([args.num_class,args.feature_dim])
        for idx,p in enumerate(proto):
            p = p.data.detach().cpu().numpy()
            c = lab[idx]
            episodic_proto[c] += p
        episodic_proto = episodic_proto/(args.shot+args.query)
        # compute the metrics
        acc = accuracy(outputs, lab)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item(),acc]
        recoder.update(vals)
        avg_proto.update(episodic_proto)

        # logging
        if i==0 or i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(trainloader))
            # Reset average meters 
            recoder.reset()        

    global_proto = avg_proto.avg
    return global_proto