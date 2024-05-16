import os
import pickle
from dataset.Database import Database, Database_pretict, Database_teech
import torch


# 数据加载器及采样器

def generate_dataset(args):
    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    tr_keys = splits[args.fold]['train']
    val_keys = splits[args.fold]['val']
    test_keys = splits[args.fold]['test']
    
    num_folds = len(splits) 
    print(num_folds)
    
    if args.tr_size < len(tr_keys):
        tr_keys = tr_keys[0:args.tr_size]
    print(tr_keys)
    print(val_keys)
    print(test_keys)
    
    # args.img_size = 224
    train_ds = Database(keys=tr_keys, mode='train', args=args)
    val_ds = Database(keys=val_keys, mode='val', args=args)
    test_ds = Database(keys=test_keys, mode='test', args=args)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader, test_loader


def generate_teechdataset(args):
    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    tr_keys = splits[args.fold]['train']
    val_keys = splits[args.fold]['val']
    test_keys = splits[args.fold]['test']

    num_folds = len(splits)
    print(num_folds)

    if args.tr_size < len(tr_keys):
        tr_keys = tr_keys[0:args.tr_size]
    print(tr_keys)
    print(val_keys)
    print(test_keys)

    # args.img_size = 224
    train_ds = Database_teech(keys=tr_keys, mode='train', args=args)
    val_ds = Database_teech(keys=val_keys, mode='val', args=args)
    test_ds = Database_teech(keys=test_keys, mode='test', args=args)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader

def generate_test_loader(key, args):
    # args.img_size = 224
    test_ds = Database(keys=key, mode='val', args=args)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    return test_loader


def generate_predict_loader(key, args, size):
    # args.img_size = 224
    test_ds = Database_pretict(keys=key, args=args, size=size )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    return test_loader
