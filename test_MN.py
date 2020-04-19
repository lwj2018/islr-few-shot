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
from utils.testUtils import test_mn
from torch.utils.tensorboard import SummaryWriter
from utils.dataUtils import getValloader
from utils.metricUtils import account_mean_and_std
from Arguments import Arguments
from utils.Averager import AverageMeter
from datasets.samplers import PretrainSampler
from datasets.CSL_Isolated_Openpose import CSL_Isolated_Openpose
from datasets.CSL_Isolated_Openpose_drop40 import CSL_Isolated_Openpose2

# Hyper params 
epochs = 100
learning_rate = 1e-5
batch_size = 8
# Options
shot = 5
dataset = 'isl'
store_name = 'test_' + dataset + '_MN' + '_%dshot'%(shot)
summary_name = 'runs/' + store_name
checkpoint = '/home/liweijie/projects/islr-few-shot/checkpoint/20200419_isl_MN_5shot_best.pth.tar'
log_interval = 20
device_list = '1'
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
trainset = CSL_Isolated_Openpose('trainvaltest')
train_sampler = PretrainSampler(trainset.label, args.shot, args.n_base, batch_size)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                        num_workers=num_workers, pin_memory=True)
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
model = MN(model_cnn,lstm_input_size=args.feature_dim,train_way=args.train_way,test_way=args.test_way,\
    shot=args.shot,query=args.query,query_val=args.query_val).to(device)

# Resume model
if checkpoint is not None:
    start_epoch, best_acc = resume_model(model, checkpoint)

global_saved = 1
if global_saved == 0:
    # Calculate global proto
    global_proto = np.zeros([args.num_class,args.feature_dim])
    for idx, batch in enumerate(train_loader):
        print('%d/%d'%(idx,len(train_loader)))
        # get the data and labels
        data,lab = [_.to(device) for _ in batch]
        proto = model.baseModel(data)
        for idx,p in enumerate(proto):
            p = p.data.detach().cpu().numpy()
            c = lab[idx]
            global_proto[c] += p
    global_proto[:args.n_base] = global_proto[:args.n_base] / args.n_reserve
    global_proto[args.n_base:] = global_proto[args.n_base:] / args.shot
    global_proto = torch.Tensor(global_proto)
    # Save
    torch.save(global_proto, '%s/%s_gproto.pth.tar' % (model_path, store_name))
else:
    # Load
    global_proto = torch.load('%s/%s_gproto.pth.tar' % (model_path, store_name))
global_proto = global_proto.to(device)

# Create loss criterion & optimizer
criterion = nn.CrossEntropyLoss()

# Start Test
print("Test Started".center(60, '#'))
for epoch in range(start_epoch, start_epoch+1):
    # Test the model
    acc = test_mn(model,global_proto,criterion,val_loader,device,epoch,log_interval,writer,args)
    print('Batch accu_a on isl: {:.3f}'.format(acc))
    acc = test_mn(model,global_proto,criterion,val_loader2,device,epoch,log_interval,writer,args)
    print('Batch accu_b on isl: {:.3f}'.format(acc))
    acc = test_mn(model,global_proto,criterion,val_loader3,device,epoch,log_interval,writer,args)
    print('Batch accu_n on isl: {:.3f}'.format(acc))

print("Test Finished".center(60, '#'))