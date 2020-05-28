import torch
import torch.nn.functional as F
import numpy
import time
from utils.metricUtils import *
from utils.Averager import AverageMeter
from utils.Recorder import Recorder
from utils.fewshotUtils import create_nshot_task_label
from models.MAML import replace_grad
from collections import OrderedDict
import pandas as pd

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
        with torch.no_grad():
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

def eval_gcr(model, criterion,
          valloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss1 = AverageMeter()
    avg_loss2 = AverageMeter()
    avg_acc1 = AverageMeter()
    avg_acc2 = AverageMeter()
    statistic = []
    # Create recorder
    averagers = [avg_loss1, avg_loss2, avg_acc1, avg_acc2]
    names = ['val loss1','val loss2','val acc1','val acc2']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        with torch.no_grad():
            # measure data loading time
            recoder.data_tok()

            # get the inputs and labels
            data, lab = [_.to(device) for _ in batch]

            # forward
            p = args.shot * args.test_way
            data_shot = data[:p]
            data_query = data[p:]

            logits, label, logits2, gt = \
                    model(data_shot,data_query,lab,mode='eval')
            # compute the loss
            loss, loss1, loss2 = criterion(logits, label, logits2, gt)
            # print('logits: {}'.format(logits))
            # print('out: {}'.format(logits.argmax(-1)))
            # print('label: {}'.format(label))

            # compute the metrics
            acc1 = accuracy(logits, label)[0]
            acc2 = accuracy(logits2, gt)[0]

            # measure elapsed time
            recoder.tok()
            recoder.tik()
            recoder.data_tik()

        # update average value & account statistic
        vals = [loss1.item(),loss2.item(),acc1,acc2]
        recoder.update(vals)
        statistic.append(acc1.data.cpu().numpy())

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')

    return recoder.get_avg('val acc1'), numpy.array(statistic)

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
        with torch.no_grad():
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
        with torch.no_grad():
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
        with torch.no_grad():
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

def test_mn(model, global_proto, criterion,
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
        with torch.no_grad():
            # measure data loading time
            recoder.data_tok()

            # get the inputs and labels
            data, lab = [_.to(device) for _ in batch]

            # forward
            y_pred = model.gfsl_test(global_proto,data)
            # compute the loss
            loss = criterion(y_pred, lab)

            # compute the metrics
            acc = accuracy(y_pred, lab)[0]

            # measure elapsed time
            recoder.tok()
            recoder.tik()
            recoder.data_tik()

        # update average value & account statistic
        vals = [loss.item(),acc]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Test')

    return recoder.get_avg('val acc')

def eval_maml(model, criterion,
          valloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    # Create recorder
    averagers = [avg_loss, avg_acc]
    names = ['val loss','val acc']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    # Settings
    create_graph = (True if args.order == 2 else False)
    for i, batch in enumerate(valloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        # data = data.view( ((args.shot+args.query),args.train_way) + data.size()[-3:] )
        # data = data.permute(1,0,2,3,4).contiguous()
        # data = data.view( (-1,) + data.size()[-3:] )
        p = args.shot * args.test_way
        data_shot = data[:p]
        data_query = data[p:]
        data_shape = data.size()[-3:]


        # Create a fast model using the current meta model weights
        fast_weights = OrderedDict(model.named_parameters())

        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(args.inner_train_steps):
            # Perform update of model weights
            y = create_nshot_task_label(args.test_way, args.shot).to(device)
            logits = model.functional_forward(data_shot, fast_weights)
            loss = criterion(logits, y)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            # Update weights manually
            fast_weights = OrderedDict(
                (name, param - args.inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
        
        # Do a pass of the model on the validation data from the current task
        y = create_nshot_task_label(args.test_way, args.query_val).to(device)
        logits = model.functional_forward(data_query, fast_weights)
        loss = criterion(logits, y)
        loss.backward(retain_graph=True)

        # Get post-update accuracies
        y_pred = logits.softmax(-1)
        acc = accuracy(y_pred, y)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item(),acc]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')

    return recoder.get_avg('val acc')

def evaluate_confusion_matrix(model, criterion,
          valloader, device, epoch, 
          log_interval, writer, args, relation, name,
          category_space='novel',
          base_class=400,novel_class=100):
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
    num_class = base_class + novel_class
    cmat = numpy.zeros([num_class,num_class])
    for i, batch in enumerate(valloader):
        with torch.no_grad():
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

            # compute the confusion matrix
            predict = logits.argmax(-1)
            for p,g in zip(predict,lab):
                cmat[g,p] += 1
            
            # measure elapsed time
            recoder.tok()
            recoder.tik()
            recoder.data_tik()

        # update average value
        vals = [loss.item(),acc]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Test')

    # normalize & print confusion matrix
    cmat = cmat / cmat.sum(1)
    if category_space == 'novel':
        cmat = cmat[base_class:,base_class:]
    elif category_space == 'base':
        cmat = cmat[:base_class,:base_class]
    df = pd.DataFrame(cmat)
    df.to_csv(name)

    return recoder.get_avg('val acc')