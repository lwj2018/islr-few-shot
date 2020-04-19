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

        optimizer.zero_grad()
        # forward
        outputs = model(data)

        # compute the loss
        loss = criterion(outputs,lab)

        # backward & optimize
        loss.backward()
        optimizer.step()

        # Calculate global proto
        proto = model.get_feature(data)
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

def train_gcr(model, criterion,
          optimizer, optimizer_cnn,
          trainloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss1 = AverageMeter()
    avg_loss2 = AverageMeter()
    avg_acc1 = AverageMeter()
    avg_acc2 = AverageMeter()
    # Create recorder
    averagers = [avg_loss1, avg_loss2, avg_acc1, avg_acc2]
    names = ['train loss1','train loss2','train acc1','train acc2']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set trainning mode
    model.train()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(trainloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        p = args.shot * args.train_way
        data_shot = data[:p]
        data_query = data[p:]

        logits, label, logits2, gt = \
                model(data_shot,data_query,lab)
        # compute the loss
        loss, loss1, loss2 = criterion(logits, label, logits2, gt)

        # backward & optimize
        optimizer.zero_grad()
        optimizer_cnn.zero_grad()
        loss.backward()
        if epoch > 45:
            optimizer_cnn.step()
        optimizer.step()

        # compute the metrics
        acc1 = accuracy(logits, label)[0]
        acc2 = accuracy(logits2, gt)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss1.item(),loss2.item(),acc1,acc2]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(trainloader))
            # Reset average meters 
            recoder.reset()

def train_gcr_relation(model, criterion,
          optimizer, optimizer_cnn,
          trainloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss1 = AverageMeter()
    avg_loss2 = AverageMeter()
    avg_loss3 = AverageMeter()
    avg_acc1 = AverageMeter()
    avg_acc2 = AverageMeter()
    avg_acc3 = AverageMeter()
    # Create recorder
    averagers = [avg_loss1, avg_loss2, avg_loss3, avg_acc1, avg_acc2, avg_acc3]
    names = ['train loss1','train loss2','train loss3','train acc1','train acc2','train acc3']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set trainning mode
    model.train()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(trainloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        p = args.shot * args.train_way
        data_shot = data[:p]
        data_query = data[p:]

        logits, label, logits2, gt, logits3, gt3 = \
                model(data_shot,data_query,lab)
        # compute the loss
        loss, loss1, loss2, loss3 = criterion(logits, label, logits2, gt, logits3, gt3)

        # backward & optimize
        optimizer.zero_grad()
        optimizer_cnn.zero_grad()
        loss.backward()
        optimizer_cnn.step()
        optimizer.step()

        # compute the metrics
        acc1 = accuracy(logits, label)[0]
        acc2 = accuracy(logits2, gt)[0]
        acc3 = accuracy(logits3, gt3)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss1.item(),loss2.item(),loss3.item(),acc1,acc2,acc3]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(trainloader))
            # Reset average meters 
            recoder.reset()

def train_mn_pn(model, criterion,
          optimizer,
          trainloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    # Create recorder
    averagers = [avg_loss, avg_acc]
    names = ['train loss','train acc']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set trainning mode
    model.train()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(trainloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        p = args.shot * args.train_way
        data_shot = data[:p]
        data_query = data[p:]

        y_pred, label = model(data_shot,data_query)
        # print('lab: {}'.format(lab.view((args.shot+args.query),args.train_way)[0]))
        # compute the loss
        loss = criterion(y_pred, label)
        # print('y_pred: {}'.format(y_pred))
        # print('label: {}'.format(label))

        # backward & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute the metrics
        acc = accuracy(y_pred, label)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item(),acc]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(trainloader))
            # Reset average meters 
            recoder.reset()
