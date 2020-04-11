import os.path as osp
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.CSL_Isolated_Openpose import CSL_Isolated_Openpose
from models.HCN import hcn
from utils.ioUtils import *
from utils.trainUtils import train_cnn
from utils.testUtils import eval_cnn
from torch.utils.tensorboard import SummaryWriter
from datasets.samplers import PretrainSampler
from Arguments import Arguments

# Hyper params 
epochs = 2000
learning_rate = 1e-5
batch_size = 8
# Options
shot = 5
dataset = 'isl'
store_name = 'HCN'
gproto_name = 'global_proto'
checkpoint = '/home/liweijie/projects/SLR/checkpoint/20200315_82.106_HCN_isolated_best.pth.tar'
log_interval = 100
device_list = '1'
num_workers = 8
model_path = "./checkpoint"

best_acc = 0.00
start_epoch = 0

# Get args
args = Arguments(shot,dataset)
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join('runs/cnn', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Prepare dataset & dataloader
trainset = CSL_Isolated_Openpose('trainvaltest')
train_sampler = PretrainSampler(trainset.label, args.shot, args.n_base, batch_size)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                        num_workers=num_workers, pin_memory=True)
print('Total size of the train set: %d'%(len(train_loader)))
valset = CSL_Isolated_Openpose('trainvaltest')
val_sampler = PretrainSampler(valset.label, args.shot, args.n_base, batch_size)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                        num_workers=num_workers, pin_memory=True)
model = hcn(args.num_class).to(device)
# Resume model
if checkpoint is not None:
    start_epoch, best_acc = resume_model(model, checkpoint)
# Create loss criterion & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Start training
print("Training Started".center(60, '#'))
for epoch in range(start_epoch, epochs):
    # Train the model
    global_proto = train_cnn(model, criterion, optimizer, train_loader, device, epoch, log_interval, writer, args)
    # Eval the model
    acc = eval_cnn(model, criterion, val_loader, device, epoch, log_interval, writer, args)
    # Save model
    # remember best acc and save checkpoint
    is_best = acc>best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best': best_acc
    }, is_best, model_path, store_name)
    save_checkpoint(global_proto, is_best, model_path, gproto_name)
    print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

print("Training Finished".center(60, '#'))