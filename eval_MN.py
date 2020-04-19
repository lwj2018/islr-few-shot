import os.path as osp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.gcrHCN import gcrHCN
from models.MN import MN
from utils.ioUtils import *
from utils.trainUtils import train_mn_pn
from utils.testUtils import eval_mn_pn
from torch.utils.tensorboard import SummaryWriter
from utils.dataUtils import getValloader
from utils.metricUtils import account_mean_and_std
from Arguments import Arguments

# Hyper params 
epochs = 100
learning_rate = 1e-5
# Options
shot = 1
dataset = 'isl'
store_name = 'eval_' + dataset + '_MN' + '_%dshot'%(shot)
summary_name = 'runs/' + store_name
checkpoint = '/home/liweijie/projects/islr-few-shot/checkpoint/isl_MN_1shot_best.pth.tar'
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
val_loader = getValloader(dataset, args)

model_cnn = gcrHCN().to(device)
model = MN(model_cnn,lstm_input_size=args.feature_dim,train_way=args.train_way,test_way=args.test_way,\
    shot=args.shot,query=args.query,query_val=args.query_val).to(device)

# Resume model
if checkpoint is not None:
    start_epoch, best_acc = resume_model(model, checkpoint)

# Create loss criterion & optimizer
criterion = nn.CrossEntropyLoss()

# Start Evaluation
print("Evaluation Started".center(60, '#'))
for epoch in range(start_epoch, start_epoch+1):
    # Eval the model
    acc, statistic = eval_mn_pn(model,criterion,val_loader,device,epoch,log_interval,writer,args)
    mean, std = account_mean_and_std(statistic)
    print('Batch acc on isl: {:.3f}'.format(acc))

print("Evaluation Finished".center(60, '#'))