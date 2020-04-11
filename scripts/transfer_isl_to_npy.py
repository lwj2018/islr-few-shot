# Util function
def content_to_mat(content):
        mat = []
        for i in range(len(content)):
            if "Body" in content[i]:
                for j in range(25):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)
            elif "Face" in content[i]:
                for j in range(70):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)

            elif "Left" in content[i]:
                for j in range(21):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)

            elif "Right" in content[i]:
                for j in range(21):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)
                break

        mat = np.array(mat)
        # 第三维是置信度，不需要
        mat = mat[:,0:2]
        return mat

def load_data(path):
        file_list = os.listdir(path)
        file_list.sort()
        mat = []
        for i,file in enumerate(file_list):
            # 第一帧有问题，先排除
            if i>0:
                filename =  osp.join(path,file)
                f = open(filename,"r")
                content = f.readlines()
                try:
                    mat_i = content_to_mat(content)
                    mat.append(mat_i)
                except:
                    print("can not convert this file to mat: "+filename)
        mat = np.array(mat)
        end = time.time()
        mat = mat.astype(np.float32)
        return mat

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


import os
import os.path as osp
import numpy as np
import time
skeleton_root = '/home/liweijie/Data/skeletons_dataset'
new_root = '/home/liweijie/Data/skeletons_dataset_npy'
create_path(new_root)
csv_file = '../csv/trainvaltest.csv'
content = open(csv_file,'r').readlines()[1:]

for i,record in enumerate(content):
    print('%d/%d'%(i,len(content)))
    # read data & transfer to mat
    name, label = record.strip().split(',')
    path = osp.join(skeleton_root,name)
    mat = load_data(path)
    # save mat
    fname = osp.join(new_root,name)
    subfolder1 = '/'.join(fname.split('/')[:-1])
    create_path(subfolder1)
    subfolder2 = '/'.join(fname.split('/')[:-2])
    create_path(subfolder2)
    np.save(fname,mat)

