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
from models.PN import PN
from models.gcrHCN import gcrHCN
from utils.ioUtils import *
from utils.trainUtils import train_cnn
from utils.testUtils import eval_cnn
from torch.utils.tensorboard import SummaryWriter
from datasets.samplers import PretrainSampler
from Arguments import Arguments
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Hyper params
batch_size = 8
# Options
shot = 5
dataset = 'isl'
model_name = 'PN'
store_name = model_name
gproto_name = 'global_proto'
log_interval = 100
device_list = '0'
num_workers = 8
out_path = "./out"
create_path(out_path)
if model_name == 'HCN':
    checkpoint = '/home/liweijie/projects/islr-few-shot/checkpoint/20200412_HCN_best.pth.tar'
elif model_name == 'PN':
    checkpoint = '/home/liweijie/projects/islr-few-shot/checkpoint/20200419_isl_PN_5shot_best.pth.tar'

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
train_loader = DataLoader(dataset=trainset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, shuffle=True)
print('Total size of the train set: %d'%(len(train_loader)))
if model_name == 'HCN':
    model = hcn(args.num_class).to(device)
    start_epoch, best_acc = resume_model(model, checkpoint)
elif model_name == 'PN':
    model_cnn = gcrHCN().to(device)
    model = PN(model_cnn,lstm_input_size=args.feature_dim,train_way=args.train_way,test_way=args.test_way,\
        shot=args.shot,query=args.query,query_val=args.query_val).to(device)
    start_epoch, best_acc = resume_model(model, checkpoint)

savename_x = osp.join(out_path,store_name+'_X.npy')
savename_y = osp.join(out_path,store_name+'_Y.npy')
saved = 0
if osp.exists(savename_x) and osp.exists(savename_y):
    saved = 1
# Account & Save
if saved == 0:
    sample_size = 1000
    X = []
    Y = []
    for idx, batch in enumerate(train_loader):
        if idx >= sample_size: break
        print('%d/%d'%(idx,sample_size))
        # get the data and labels
        data,lab = [_.to(device) for _ in batch]
        proto = model.get_feature(data)
        proto = proto.data.cpu().numpy()
        lab = lab.data.cpu().numpy()
        X.extend(proto)
        Y.extend(lab)
    X = np.array(X)
    Y = np.array(Y)
    np.save(savename_x,X)
    np.save(savename_y,Y)
# Load
else:
    X = np.load(savename_x)
    Y = np.load(savename_y)
# Tsne fit & Plot
start = time.time()
print('Fitting...')
X_embedded = TSNE(n_components=2).fit_transform(X)
end = time.time()
print('Fit cost %.3f s'%(end-start))
cmap = plt.cm.get_cmap('hsv')
plt.scatter(X_embedded[:,0],X_embedded[:,1],s=5,c=Y,cmap=cmap,alpha=0.7)
plt.colorbar()
plt.savefig(osp.join(out_path,store_name))



