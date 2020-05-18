from torch.utils.data import DataLoader
from datasets.CSL_Isolated_Openpose import CSL_Isolated_Openpose
from datasets.samplers import CategoriesSampler_train, CategoriesSampler_val

def getDataloader(dataset,args):
    trainset = CSL_Isolated_Openpose('trainvaltest',is_aug=True)
    train_sampler = CategoriesSampler_train(trainset.label, 100,
                            args.train_way, args.shot, args.query, args.n_base, args.n_reserve)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                            num_workers=args.num_workers, pin_memory=True)
    valset = CSL_Isolated_Openpose('test')
    val_sampler = CategoriesSampler_val(valset.label, 100,
                            args.test_way, args.shot, args.query_val)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader

def getValloader(dataset,args):
    valset = CSL_Isolated_Openpose('test')
    val_sampler = CategoriesSampler_val(valset.label, 600,
                            args.test_way, args.shot, args.query_val)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)
    return val_loader