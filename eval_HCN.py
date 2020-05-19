import os.path as osp
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.CSL_Isolated_Openpose_drop40 import CSL_Isolated_Openpose2
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
checkpoint = '/home/liweijie/projects/islr-few-shot/checkpoint/HCN_5shot_f32_best.pth.tar'
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
valset = CSL_Isolated_Openpose2('trainvaltest')
val_loader = DataLoader(dataset=valset, batch_size = 8,
                        num_workers=8, pin_memory=True, shuffle=True)
valset2 = CSL_Isolated_Openpose2('trainval')
val_loader2 = DataLoader(dataset=valset2, batch_size = 8,
                        num_workers=8, pin_memory=True, shuffle=True)
valset3 = CSL_Isolated_Openpose2('test')
val_loader3 = DataLoader(dataset=valset3, batch_size = 8,
                        num_workers=8, pin_memory=True, shuffle=True)
print('Total size of the val set: %d'%(len(val_loader)))
model = hcn(args.num_class,f_dim=args.feature_dim).to(device)
# Resume model
if checkpoint is not None:
    start_epoch, best_acc = resume_model(model, checkpoint)
# Create loss criterion & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Start Evaluation
print("Evaluation Started".center(60, '#'))
for epoch in range(start_epoch, start_epoch+1):
    # Eval the model
    acc = eval_cnn(model,criterion,val_loader3,device,epoch,log_interval,writer,args)
    print('Batch accu_n on isl: {:.3f}'.format(acc))
    acc = eval_cnn(model,criterion,val_loader,device,epoch,log_interval,writer,args)
    print('Batch accu_a on isl: {:.3f}'.format(acc))
    acc = eval_cnn(model,criterion,val_loader2,device,epoch,log_interval,writer,args)
    print('Batch accu_b on isl: {:.3f}'.format(acc))

print("Evaluation Finished".center(60, '#'))