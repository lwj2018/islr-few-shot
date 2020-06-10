import os.path as osp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.CSL_Isolated_Openpose_drop40 import CSL_Isolated_Openpose2
from models.GCR_base import GCR_base
from models.gcrHCN_origin import gcrHCN
from models.Hallucinator import Hallucinator
from utils.ioUtils import *
from utils.critUtils import loss_for_gcr
from utils.testUtils import test_gcr
from torch.utils.tensorboard import SummaryWriter
from utils.dataUtils import getDataloader
from Arguments import Arguments

# Options
shot = 5
dataset = 'isl'
store_name = 'test_' + dataset + '_GCR_base' + '_%dshot'%(shot)
summary_name = 'runs/' + store_name
checkpoint = '/home/liweijie/projects/islr-few-shot/checkpoint/isl_GCR_base_5shot_best.pth.tar'#5-shot
# checkpoint = '/home/liweijie/projects/few-shot/checkpoint/20200404_miniImage_GCR_r_1shot_best.pth.tar'#1-shot
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
valset = CSL_Isolated_Openpose2('trainvaltest')
val_loader = DataLoader(dataset=valset, batch_size = 8,
                        num_workers=8, pin_memory=True, shuffle=True)
valset2 = CSL_Isolated_Openpose2('trainval')
val_loader2 = DataLoader(dataset=valset2, batch_size = 8,
                        num_workers=8, pin_memory=True, shuffle=True)
valset3 = CSL_Isolated_Openpose2('test')
val_loader3 = DataLoader(dataset=valset3, batch_size = 8,
                        num_workers=8, pin_memory=True, shuffle=True)

model_cnn = gcrHCN().to(device)
# model_gen = Hallucinator(args.feature_dim).to(device)
model = GCR_base(model_cnn,train_way=args.train_way,\
    test_way=args.test_way, shot=args.shot,query=args.query,query_val=args.query_val,f_dim=args.feature_dim).to(device)
# Resume model
if checkpoint is not None:
    start_epoch, best_acc = resume_gcr_model(model, checkpoint, args.n_base)

# Create loss criterion
criterion = nn.CrossEntropyLoss()

# Start Test
print("Test Started".center(60, '#'))
for epoch in range(start_epoch, start_epoch+1):
    acc = test_gcr(model,criterion,val_loader3,device,epoch,log_interval,writer,args,model.relation1)
    print('Batch accu_n on isl: {:.3f}'.format(acc))
    acc = test_gcr(model,criterion,val_loader,device,epoch,log_interval,writer,args,model.relation1)
    print('Batch accu_a on isl: {:.3f}'.format(acc))
    acc = test_gcr(model,criterion,val_loader2,device,epoch,log_interval,writer,args,model.relation1)
    print('Batch accu_b on isl: {:.3f}'.format(acc))

print("Test Finished".center(60, '#'))