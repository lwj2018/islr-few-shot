import torch
import torch.nn.functional as F
import numpy
import time
from utils.metricUtils import *
from utils.Averager import AverageMeter
from utils.Recorder import Recorder
from utils.fewshotUtils import create_nshot_task_label

def eval_cnn(model, criterion, valloader, 
        device, epoch, log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    averagers = [losses, avg_acc]
    names = ['val loss','val acc']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        # measure data loading time
        recoder.data_tok()

        # get the data and labels
        data,lab = [_.to(device) for _ in batch]

        p = args.shot * args.test_way
        data_shot = data[:p]
        data_query = data[p:]
        input = torch.cat([data_shot,data_query],0)

        # forward
        outputs = model(input)

        # compute the loss
        loss = criterion(outputs,lab)

        # compute the metrics
        acc = accuracy(outputs, lab)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item(),acc]
        recoder.update(vals)

        # logging
        if i==0 or i % log_interval == log_interval-1 or i==len(valloader)-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')
        
    return recoder.get_avg('val acc')

def eval_gcr_relation(model, criterion,
          valloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss1 = AverageMeter()
    avg_loss2 = AverageMeter()
    avg_loss3 = AverageMeter()
    avg_acc1 = AverageMeter()
    avg_acc2 = AverageMeter()
    avg_acc3 = AverageMeter()
    statistic = []
    # Create recorder
    averagers = [avg_loss1, avg_loss2, avg_loss3, avg_acc1, avg_acc2, avg_acc3]
    names = ['val loss1','val loss2','val loss3', 'val acc1','val acc2','val acc3']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        p = args.shot * args.test_way
        data_shot = data[:p]
        data_query = data[p:]

        logits, label, logits2, gt, logits3, gt3 = \
                model(data_shot,data_query,lab,mode='eval')
        # compute the loss
        loss, loss1, loss2, loss3 = criterion(logits, label, logits2, gt, logits3, gt3)
        # print('logits: {}'.format(logits))
        # print('out: {}'.format(logits.argmax(-1)))
        # print('label: {}'.format(label))

        # compute the metrics
        acc1 = accuracy(logits, label)[0]
        acc2 = accuracy(logits2, gt)[0]
        acc3 = accuracy(logits3, gt3)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value & account statistic
        vals = [loss1.item(),loss2.item(),loss3.item(),acc1,acc2,acc3]
        recoder.update(vals)
        statistic.append(acc1.data.cpu().numpy())

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')

    return recoder.get_avg('val acc3'), numpy.array(statistic)

def test_gcr(model, criterion,
          valloader, device, epoch, 
          log_interval, writer, args, relation):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    # Create recorder
    averagers = [avg_loss, avg_acc]
    names = ['val loss', 'val acc']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        proto = model.baseModel(data)
        global_set = torch.cat([model.global_base,model.global_novel])
        logits = relation(proto,global_set)
        # print('logits: ',logits.argmax(-1))
        # print('lab: ',lab)
        
        # compute the loss
        loss = criterion(logits, lab)

        # compute the metrics
        acc = accuracy(logits, lab)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item(),acc]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Test')

    return recoder.get_avg('val acc')

def eval_mn_pn(model, criterion,
          valloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    statistic = []
    # Create recorder
    averagers = [avg_loss,avg_acc]
    names = ['val loss','val acc']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        p = args.shot * args.test_way
        data_shot = data[:p]
        data_query = data[p:]

        y_pred, label = model(data_shot,data_query,mode='eval')
        # print('lab: {}'.format(lab.view((args.shot+args.query_val),args.test_way)[0]))
        # compute the loss
        loss = criterion(y_pred, label)
        # print('y_pred: {}'.format(y_pred.argmax(-1)))
        # print('label: {}'.format(label))

        # compute the metrics
        acc = accuracy(y_pred, label)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value & account statistic
        vals = [loss.item(),acc]
        recoder.update(vals)
        statistic.append(acc.data.cpu().numpy())

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')

    return recoder.get_avg('val acc'), numpy.array(statistic)
