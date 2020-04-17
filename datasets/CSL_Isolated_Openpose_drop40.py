import torch.utils.data as data

from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import os
import os.path as osp
import time
import matplotlib.pyplot as plt
import time

# Params
skeleton_root = "/home/liweijie/Data/skeletons_dataset_npy"
csv_root = '/home/liweijie/projects/islr-few-shot/csv'

class CSL_Isolated_Openpose2(data.Dataset):
    
    def __init__(self, setname, skeleton_root=skeleton_root, 
            csv_root=csv_root,
            length=32, is_normalize=True):
        self.setname = setname
        self.skeleton_root = skeleton_root
        self.csv_root = csv_root
        self.length = length
        self.is_normalize = is_normalize
        self.n_samples = 50
        self.n_drop = 40
        
        self._parse_list()

    def __getitem__(self, index):
        path = self.data[index]
        lb = self.label[index]
        # Get mat
        # The shape of mat is T x J x D
        mat = self._load_data(path)
        indices = self.get_sample_indices(mat.shape[0])    
        mat = mat[indices,:,:]
        if self.is_normalize:
            for i in range(len(mat)):
                mat[i] = self.normalize(mat[i])

        return mat, lb
        

    def __len__(self):
        return len(self.data)
        
    def _parse_list(self):
        csv_path = osp.join(self.csv_root, self.setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []

        for i,l in enumerate(lines):
            if i % self.n_samples >= self.n_drop:
                name, lb = l.split(',')
                path = osp.join(self.skeleton_root, name)
                lb = int(lb)
                data.append(path)
                label.append(lb)

        self.data = data
        self.label = label

        print('video number:%d'%(len(self.data)))

    def get_sample_indices(self,num_frames):
        # ignore the first frame
        indices = np.linspace(0,num_frames-1,self.length).astype(int)
        indices = np.clip(indices,0,num_frames-1)
        return indices
    
    def _load_data(self, path):
        start = time.time()
        mat = np.load(path+'.npy')
        end = time.time()
        # print('%.4f s'%(end-start))
        mat = mat.astype(np.float32)
        return mat

    def normalize(self,mat):
        # Shape of mat is: J x D
        max_x = np.max(mat[:,0])
        min_x = min(mat[:,0])
        max_y = np.max(mat[:,1])
        min_y = min(mat[:,1])
        center_x = (max_x+min_x)/2
        center_y = (max_y+min_y)/2
        mat = (mat-[center_x,center_y])/[(max_x-min_x)/2,(max_y-min_y)/2]
        # TEST
        # print("max_x: %.2f,min_x: %.2f,max_y: %.2f,min_y: %.2f"%(max_x,min_x,max_y,min_y))
        return mat

def min(array):
    threshold = 0.1
    min = 999999
    for x in array:
        if x>threshold and x<min:
            min = x
    return min

# Test
if __name__ == '__main__':
    # Path settings
    dataset = CSL_Isolated_Openpose('trainvaltest')
    print(dataset[1000][1])