import os.path as osp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.MAML import FewShotClassifier
from utils.ioUtils import *
from utils.trainUtils import train_maml
from utils.testUtils import eval_maml
from torch.utils.tensorboard import SummaryWriter
from utils.dataUtils import getDataloader

class Arguments:
    def __init__(self,shot,dataset):

        # Settings for 5-shot
        if shot == 5:
            self.shot = 5
            self.query = 5
            self.query_val = 5
        # Settings for 1-shot
        elif shot == 1:
            self.shot = 1
            self.query = 1
            self.query_val = 5
        
        if dataset == 'isl':
            self.channel = 3
            self.n_reserve = 40
            self.n_base = 400
            self.num_class = 500
        self.train_way = 5
        self.test_way = 5
        self.feature_dim = 1600
        # Hyper params for maml
        self.inner_train_steps = 1
        self.inner_lr = 0.4
        self.order = 1
        # Options
        self.num_workers = 8

# Hyper params 
epochs = 1000
learning_rate = 1e-4
order = 1
# Options
shot = 5
dataset = 'isl'
store_name = dataset + '_MAML' + '_%dshot'%(shot)
summary_name = 'runs/' + store_name
cnn_ckpt = None#'/home/liweijie/projects/few-shot/checkpoint/20200329/CNN_best.pth.tar'
checkpoint = '/home/liweijie/projects/islr-few-shot/checkpoint/20200505_isl_MAML_5shot_best.pth.tar'
log_interval = 20
device_list = '0'
num_workers = 8
model_path = "./checkpoint"

start_epoch = 0
best_acc = 0.00
# Get args
args = Arguments(shot,dataset)
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join(summary_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Prepare dataset & dataloader
train_loader, val_loader = getDataloader(dataset, args)

model = FewShotClassifier(args.num_class).to(device)

# Resume model
if cnn_ckpt is not None:
    resume_cnn_part(model,cnn_ckpt)
if checkpoint is not None:
    start_epoch, best_acc = resume_model(model, checkpoint)

# Create loss criterion & optimizer
criterion = nn.CrossEntropyLoss()

policies = model.get_optim_policies(learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)

best_acc = 0.0
# Start training
print("Training Started".center(60, '#'))
for epoch in range(start_epoch, start_epoch + epochs):
    # Train the model
    train_maml(model,criterion,optimizer,train_loader,device,epoch,log_interval,writer,args)
    # Eval the model
    acc = eval_maml(model,criterion,val_loader,device,epoch,log_interval,writer,args)
    # Save model
    # remember best acc and save checkpoint
    is_best = acc>best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best': best_acc
    }, is_best, model_path, store_name)
    print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

print("Training Finished".center(60, '#'))