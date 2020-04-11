import os
import torch
import shutil

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(state, is_best, model_path, name):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (model_path, name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (model_path, name),
            '%s/%s_best.pth.tar' % (model_path, name))

def load_global_proto(global_ckpt,args):
    global_proto = torch.load(global_ckpt)
    global_proto = global_proto[:args.num_class,:]
    global_proto = torch.Tensor(global_proto)
    global_base = global_proto[:args.n_base,:]
    global_base = global_base.detach().cuda()
    global_novel = global_proto[args.n_base:,:]
    global_novel = global_novel.detach().cuda()
    return global_base, global_novel

def resume_model(model, checkpoint):
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    model.load_state_dict(state_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load model from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
    return params_dict['epoch'], params_dict['best']

def resume_cnn_from_cnn_gen(model, checkpoint):
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    state_dict = {'.'.join(k.split('.')[1:]) : v for k,v in state_dict.items() if 'cnn' in k 
        and not 'fc' in k}
    model.load_state_dict(state_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load CNN from CNN GEN {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))

def resume_gen_from_cnn_gen(model, checkpoint):
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    state_dict = {'.'.join(k.split('.')[1:]) : v for k,v in state_dict.items() if 'gen' in k}
    model.load_state_dict(state_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load GEN from CNN GEN {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))

def resume_cnn_for_cnn_gen(model, checkpoint):
    model_dict = model.state_dict()
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    state_dict = {'cnn.'+k : v for k,v in state_dict.items()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load CNN part for CNN GEN {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
    return model