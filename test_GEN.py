import os.path as osp
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.CSL_Isolated_Openpose import CSL_Isolated_Openpose
from models.CNN_GEN import CNN_GEN
from utils.ioUtils import *
from utils.trainUtils import train_cnn
from utils.testUtils import eval_cnn
from torch.utils.tensorboard import SummaryWriter
from datasets.samplers import PretrainSampler   
from Arguments import Arguments


# Hyper params 
epochs = 1000
learning_rate = 1e-5
batch_size = 8
# Options
shot = 5
dataset = 'isl'
store_name = 'HCN_GEN'
gproto_name = 'global_proto'
checkpoint = '/home/liweijie/projects/islr-few-shot/checkpoint/20200412_HCN_GEN_best.pth.tar'
log_interval = 20
device_list = '1'
model_path = "./checkpoint"
num_workers = 8

best_acc = 0.00
start_epoch = 0

# Get args
args = Arguments(shot,dataset)
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join('runs/hcn_gen', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Prepare dataset & dataloader
trainset = CSL_Isolated_Openpose('trainvaltest')
train_sampler = PretrainSampler(trainset.label, args.shot, args.n_base, batch_size)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                        num_workers=num_workers, pin_memory=True)
valset = CSL_Isolated_Openpose('trainvaltest')
val_sampler = PretrainSampler(valset.label, args.shot, args.n_base, batch_size)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                        num_workers=num_workers, pin_memory=True)
model = CNN_GEN(out_dim=args.num_class, f_dim=args.feature_dim).to(device)
# Resume model
if checkpoint is not None:
    start_epoch, best_acc = resume_model(model, checkpoint)
# Create loss criterion & optimizer
criterion = nn.CrossEntropyLoss()

print(model.gen.fc1.weight.size())

# Start test
print("Test Started".center(60, '#'))
for epoch in range(start_epoch, epochs):
    # Eval the model
    acc = eval_cnn(model, criterion, val_loader, device, epoch, log_interval, writer, args)

print("Test Finished".center(60, '#'))