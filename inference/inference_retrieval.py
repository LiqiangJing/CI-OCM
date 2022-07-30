"""
inference for retrieval task
"""

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

import torch
from tqdm import tqdm

from config import args_info, device
from models import CF
from utils import DataLoader, BenchmarkDataset
from utils.metrics import MRR_HR


def inference(model, retrieval_loader, rand_num):
    # record
    print("inference....")
    HR_ks = (5, 10, 40)

    model.eval()
    with torch.no_grad():
        scores = []
        tqdm_retrieval_loader = tqdm(retrieval_loader)
        tqdm_retrieval_loader.set_description("Inference Retrieval Task")
        for auc_batch in tqdm_retrieval_loader:
            auc_batch = auc_batch.to(device)
            auc_score = model.test_retrieval(auc_batch, ranking_neg_num=rand_num)
            scores.append(auc_score.cpu())

        scores = torch.cat(scores).numpy()
        res_metrics = MRR_HR(scores, HR_ks=HR_ks)

    for idx, kk in enumerate(HR_ks):
        print("HR@{}:{:.3f} ".format(kk, res_metrics[1][idx]), end=' ')
    print("")


def main(args):
    # data preparation
    BenchmarkDataset.init(args)
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.num_worker,
                     'pin_memory': True if torch.cuda.is_available() else False}
    train_dataset = BenchmarkDataset('train')
    retrieval_dataset = BenchmarkDataset('test').test_retrieval()
    retrieval_loader = DataLoader(retrieval_dataset, **loader_kwargs)

    # define model
    model = CF(num_node_features=args.hid, num_cotpye=train_dataset.num_cotpye,
               depth=args.gd, nhead=args.nhead, dim_feedforward=args.fdim,
               num_layers=arg.nlayer, num_category=train_dataset.num_semantic_category).to(device)

    if os.path.exists(args.test):
        print("===> load ckpt:", args.test)
        ckpt = torch.load(args.test, map_location=torch.device('cuda'))
        model.load_state_dict(ckpt["state_dict"])
        inference(model, retrieval_loader, args.retrieval_neg_num)
    else:
        print("===> fail to load ckpt:", args.test)


if __name__ == '__main__':
    arg = args_info()
    arg.data_dir = "../data"
    arg.batch_size = 1
    main(arg)
