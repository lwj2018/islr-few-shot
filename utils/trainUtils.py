import torch
import torch.nn.functional as F
import numpy
import time
from utils.metricUtils import *
from utils.Averager import AverageMeter
from utils.Recorder import Recorder
from utils.fewshotUtils import create_nshot_task_label
from collections import OrderedDict
from models.MAML import replace_grad
from collections import OrderedDict

def train_cnn(model, criterion, optimizer, trainloader, 
        device, epoch, log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    global_proto = numpy.zeros([args.num_class,args.feature_dim])
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

        # Account global proto
        proto = model.get_feature(data)
        for idx,p in enumerate(proto):
            p = p.data.detach().cpu().numpy()
            c = lab[idx]
            global_proto[c] += p
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
        if i==0 or i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(trainloader))
            # Reset average meters 
            recoder.reset()        

    global_proto[:args.n_base] = global_proto[:args.n_base] / args.n_reserve
    global_proto[args.n_base:] = global_proto[args.n_base:] / args.shot
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
    names = ['train loss','train loss2','train loss3','train acc','train acc2','train acc3']
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

def train_maml(model, criterion,
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
    # Settings
    create_graph = (True if args.order == 2 else False)
    task_gradients = []
    task_losses = []
    for i, batch in enumerate(trainloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        # data = data.view( ((args.shot+args.query),args.train_way) + data.size()[-3:] )
        # data = data.permute(1,0,2,3,4).contiguous()
        # data = data.view( (-1,) + data.size()[-3:] )
        p = args.shot * args.train_way
        data_shot = data[:p]
        data_query = data[p:]
        data_shape = data_shot.size()[-3:]


        # Create a fast model using the current meta model weights
        fast_weights = OrderedDict(model.named_parameters())

        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(args.inner_train_steps):
            # Perform update of model weights
            y = create_nshot_task_label(args.train_way, args.shot).to(device)
            logits = model.functional_forward(data_shot, fast_weights)
            loss = criterion(logits, y)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            # Update weights manually
            fast_weights = OrderedDict(
                (name, param - args.inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
        
        # Do a pass of the model on the validation data from the current task
        y = create_nshot_task_label(args.train_way, args.query).to(device)
        logits = model.functional_forward(data_query, fast_weights)
        loss = criterion(logits, y)
        loss.backward(retain_graph=True)

        # Get post-update accuracies
        y_pred = logits.softmax(-1)
        acc = accuracy(y_pred, y)[0]

        # Accumulate losses and gradients
        task_losses.append(loss)
        gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
        named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
        task_gradients.append(named_grads)

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

    if args.order == 1:
        sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                  for k in task_gradients[0].keys()}
        hooks = []
        for name, param in model.named_parameters():
            hooks.append(
                param.register_hook(replace_grad(sum_task_gradients, name))
            )

        model.train()
        optimizer.zero_grad()
        # Dummy pass in order to create `loss` variable
        # Replace dummy gradients with mean task gradients using hooks
        logits = model(torch.zeros((args.train_way, ) + data_shape).to(device, dtype=torch.float))
        loss = criterion(logits, create_nshot_task_label(args.train_way, 1).to(device))
        loss.backward()
        optimizer.step()

        for h in hooks:
            h.remove()
    
    elif args.order == 2:
        model.train()
        optimizer.zero_grad()
        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_loss.backward()
        optimizer.step()


