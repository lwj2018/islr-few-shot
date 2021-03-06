import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.GCR_b import GCR_b
from models.gcrHCN_origin import gcrHCN
from models.Hallucinator import Hallucinator
from utils.ioUtils import *
from utils.critUtils import loss_for_gcr, loss_for_gcr_relation
from utils.trainUtils import train_gcr_relation
from utils.testUtils import eval_gcr
from torch.utils.tensorboard import SummaryWriter
from utils.dataUtils import getValloader
from Arguments import Arguments

# Hyper params 
epochs = 2000
learning_rate = 1e-3
# Options
shot = 1
dataset = 'isl'
# Get args
args = Arguments(shot,dataset)
store_name = 'eval_' + dataset + '_GCR_b' + '_%dshot'%(args.shot)
summary_name = 'runs/' + store_name
checkpoint = '/home/liweijie/projects/islr-few-shot/checkpoint/isl_GCR_b_1shot_best.pth.tar'
log_interval = 20
device_list = '0'
model_path = "./checkpoint"

start_epoch = 0
best_acc = 0.00
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join(summary_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Prepare dataset & dataloader
val_loader = getValloader(dataset,args)

model_cnn = gcrHCN().to(device)
model = GCR_b(model_cnn,train_way=args.train_way,\
    test_way=args.test_way, shot=args.shot,query=args.query,query_val=args.query_val,f_dim=args.feature_dim).to(device)
# Resume model
if checkpoint is not None:
    start_epoch, best_acc = resume_gcr_model(model, checkpoint, args.n_base)

# Create loss criterion & optimizer
criterion = loss_for_gcr()

# Start Evaluation
print("Evaluation Started".center(60, '#'))
for epoch in range(start_epoch, start_epoch+1):
    # Eval the model
    acc,_ = eval_gcr(model,criterion,val_loader,device,epoch,log_interval,writer,args)
    print('Batch acc on isl: {:.3f}'.format(acc))

print("Evaluation Finished".center(60, '#'))