"""
inference for compatibility modeling task and fill-in-the-blank task
"""

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import device, args_info
from models import CF
from utils import DataLoader, BenchmarkDataset


def inference(model, auc_loader, fitb_loader):
    model.eval()
    with torch.no_grad():
        scores, labels = [], []
        tqdm_auc_loader = tqdm(auc_loader)
        tqdm_auc_loader.set_description("Inference Compatibility Modeling Task")
        for auc_batch in tqdm_auc_loader:
            auc_batch = auc_batch.to(device)
            auc_score = model.test_auc(auc_batch)
            scores.append(auc_score.cpu())
            labels.append(auc_batch.y.cpu())

        scores = torch.cat(scores).numpy()
        labels = torch.cat(labels).numpy()
        cp_auc = roc_auc_score(labels, scores)

        fitb_right = 0
        tqdm_fitb_loader = tqdm(fitb_loader)
        tqdm_fitb_loader.set_description("Inference FITB Task")
        for fitb_batch in tqdm_fitb_loader:
            fitb_batch = fitb_batch.to(device)
            fitb_right += model.test_fitb(fitb_batch)
        fitb_acc = fitb_right / len(fitb_loader.dataset)

    return cp_auc, fitb_acc


def main(args):
    # data preparation
    BenchmarkDataset.init(args) #?
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.num_worker,
                     'pin_memory': True if torch.cuda.is_available() else False}
    train_dataset = BenchmarkDataset('train')
    auc_dataset = BenchmarkDataset('test').test_auc()
    fitb_dataset = BenchmarkDataset('test').test_fitb()
    auc_loader = DataLoader(auc_dataset, **loader_kwargs)
    fitb_loader = DataLoader(fitb_dataset, **loader_kwargs)

    # define model
    model = CF(num_node_features=args.hid, num_cotpye=train_dataset.num_cotpye,
               depth=args.gd, nhead=args.nhead, dim_feedforward=args.fdim,
               num_layers=arg.nlayer, num_category=train_dataset.num_semantic_category).to(device)

    if os.path.exists(args.test):
        print("===> load ckpt:", args.test)
        ckpt = torch.load(args.test, map_location=torch.device('cuda'))
        model.load_state_dict(ckpt["state_dict"])
        cp_auc, fitb_acc = inference(model, auc_loader, fitb_loader)
        print(f"auc: {cp_auc:.4f} fitb: {fitb_acc:.4f} ")
    else:
        print("===> fail to load ckpt:", args.test)


if __name__ == '__main__':
    arg = args_info()
    arg.data_dir = "../data"
    main(arg)
