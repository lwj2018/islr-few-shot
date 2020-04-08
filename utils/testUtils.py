import torch
import torch.nn.functional as F
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
        data_shot = data_shot[:,3:,:]
        data_query = data_query[:,3:,:]
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
        
    return recoder.get_avg('val acc')12345