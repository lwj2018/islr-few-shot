import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.GCR_base import GCR_base
from models.gcrHCN_origin import gcrHCN
from models.Hallucinator import Hallucinator
from utils.ioUtils import *
from utils.critUtils import loss_for_gcr_base
from utils.trainUtils import train_gcr_relation
from utils.testUtils import eval_gcr_relation
from torch.utils.tensorboard import SummaryWriter
from utils.dataUtils import getDataloader
from Arguments import Arguments

# Hyper params 
epochs = 20
learning_rate = 1e-3#default 1e-5
# Options
shot = 1
dataset = 'isl'
# Get args
args = Arguments(shot,dataset)
store_name = dataset + '_GCR_base' + '_%dshot'%(args.shot)
summary_name = 'runs/' + store_name
cnn_ckpt = '/home/liweijie/projects/islr-few-shot/checkpoint/20200419_HCN_best.pth.tar'
global_ckpt = '/home/liweijie/projects/islr-few-shot/checkpoint/20200419_global_proto_best.pth.tar'
cnngen_ckpt = None#'/home/liweijie/projects/islr-few-shot/checkpoint/20200412_HCN_GEN_best.pth.tar'
checkpoint = None#'/home/liweijie/projects/islr-few-shot/checkpoint/isl_GCR_5shot_best.pth.tar'
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
train_loader, val_loader = getDataloader(dataset,args)

model_cnn = gcrHCN().to(device)
# model = GCR(model_cnn,train_way=args.train_way,\
#     test_way=args.test_way, shot=args.shot,query=args.query,query_val=args.query_val,f_dim=args.feature_dim).to(device)
# Resume model
if cnn_ckpt is not None:
    resume_cnn_part(model_cnn,cnn_ckpt)
if cnngen_ckpt is not None:
    resume_cnn_from_cnn_gen(model_cnn,cnngen_ckpt)
    resume_gen_from_cnn_gen(model_gen,cnngen_ckpt)
if checkpoint is not None:
    start_epoch, best_acc = resume_gcr_model(model, checkpoint, args.n_base)
global_base, global_novel = load_global_proto(global_ckpt,args)

model = GCR_base(model_cnn,global_base=global_base,global_novel=global_novel,train_way=args.train_way,\
    test_way=args.test_way, shot=args.shot,query=args.query,query_val=args.query_val,f_dim=args.feature_dim).to(device)

# Create loss criterion & optimizer
criterion = loss_for_gcr_base()

policies = model.get_optim_policies(learning_rate)
optimizer = torch.optim.SGD(policies, momentum=0.9)
optimizer_cnn = torch.optim.SGD(model.baseModel.parameters(), lr=learning_rate,momentum=0.9)
# optimizer = torch.optim.Adam(policies)
# optimizer_cnn = torch.optim.Adam(model.baseModel.parameters())

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
lr_scheduler_cnn = torch.optim.lr_scheduler.MultiStepLR(optimizer_cnn, milestones=[30,60], gamma=0.1)

# Start training
best_acc = 0.0
print(f"Training setting: {args.test_way} way {args.shot} shot")
print("Training Started".center(60, '#'))
for epoch in range(start_epoch, start_epoch + epochs):
    # Train the model
    train_gcr_relation(model,criterion,optimizer,optimizer_cnn,train_loader,device,epoch,log_interval,writer,args)
    # Eval the model
    acc,_ = eval_gcr_relation(model,criterion,val_loader,device,epoch,log_interval,writer,args)
    # Save model
    # remember best acc and save checkpoint
    is_best = acc>best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best': best_acc,
        'global_proto': torch.cat([model.global_base,model.global_novel])
    }, is_best, model_path, store_name)
    print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))
    print('Epoch best acc: {:.3f}'.format(best_acc))

print("Training Finished".center(60, '#'))