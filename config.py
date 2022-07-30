import time
from argparse import ArgumentParser

import torch
from torch.backends import cudnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True


def args_info():
    parser = ArgumentParser(description='OCM-CF Config')
    parser.add_argument('--data-dir', type=str, default='data', help="path to dataset")
    parser.add_argument('--polyvore-split', default='nondisjoint', type=str, choices=['nondisjoint', 'disjoint'], help="version of dataset")

    parser.add_argument('-bs', dest='batch_size', type=int, default=64, help="batch size")
    parser.add_argument('-epoch', type=int, default=80)
    parser.add_argument('-lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('-j', dest='num_worker', type=int, default=6, help="number of worker")

    parser.add_argument('-hid', type=int, default=64, help="the size of image embedding")
    parser.add_argument('-gd', type=int, default=3, help="the depth of the GNN model")
    parser.add_argument('-nhead', type=int, default=8, help="the number of the Multi-head Attention")
    parser.add_argument('-nlayer', type=int, default=3, help="the number of the layer in Multi-head Attention")
    parser.add_argument('-fdim', type=int, default=256, help="the size of the FFN")
    parser.add_argument('-asp', dest="aspect", type=int, default=5, help="the number of the parallel branches")

    parser.add_argument('-re', dest='remark', type=str, default=f'exp_{int(time.time())}', help="the suffix of the experiment")
    parser.add_argument('-test', type=str, default="", help="best ckpt path")
    parser.add_argument('-retrieval_neg_num', type=int, default=499, help="the number of the negative sample in retrieval")

    args = parser.parse_args()
    _ = print("=" * 15, "args", "=" * 15), print(args), print("=" * 36)
    return args
