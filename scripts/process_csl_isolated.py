import os
import os.path as osp
import numpy
num_class = 500
n_base = 400
rearrange = numpy.random.permutation(numpy.arange(num_class))
base_class = rearrange[:n_base]
novel_class = rearrange[n_base:]
dict_file = open('/home/liweijie/Data/SLR_dataset/dictionary.txt','r')
newdict_file = open('new_dictionary.txt','w')
dict = {}
for record in dict_file.readlines():
    ind, word = record.rstrip('\n').split('\t')
    dict[int(ind)] = word
for i,ind in enumerate(rearrange):
    word = dict[ind]
    record = '%06d'%i + '\t' + '%06d'%ind + '\t' + word + '\n'
    newdict_file.write(record)

skeleton_root = "/home/liweijie/skeletons_dataset"
trainvaltest_csv = open('../csv/trainvaltest.csv','w')
trainvaltest_csv.write('foldername,label\n')
trainval_csv = open('../csv/trainval.csv','w')
trainval_csv.write('foldername,label\n')
test_csv = open('../csv/test.csv','w')
test_csv.write('foldername,label\n')
for i,c in enumerate(base_class):
    c_folder = '%06d'%c
    c_path = osp.join(skeleton_root,c_folder)
    skeleton_list = os.listdir(c_path)
    skeleton_list.sort()
    for skeleton in skeleton_list:
        skeleton_path = osp.join(c_folder,skeleton)
        record = skeleton_path + ',' + str(i) + '\n'
        trainvaltest_csv.write(record)
        trainval_csv.write(record)
for i,c in enumerate(novel_class):
    c_folder = '%06d'%c
    c_path = osp.join(skeleton_root,c_folder)
    skeleton_list = os.listdir(c_path)
    skeleton_list.sort()
    for skeleton in skeleton_list:
        skeleton_path = osp.join(c_folder,skeleton)
        record = skeleton_path + ',' + str(i+n_base) + '\n'
        trainvaltest_csv.write(record)
        test_csv.write(record)  

