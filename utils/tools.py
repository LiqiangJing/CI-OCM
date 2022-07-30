# -*- coding: utf-8 -*-
# Date: 2020/11/21 15:26

"""
tool lib
"""
__author__ = 'tianyu'

import glob
import os
import shutil
import subprocess
import time
from os import path as osp
from pathlib import Path

import torch


def get_lr(optimizer):
    r"""
    get the now learning rate
    :param optimizer:
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_dir(d_path):
    dir_path = Path(d_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def repeat_dir_name(log_path):
    if os.path.isdir(log_path):
        rep_len = len(list(glob.glob(f"{log_path}*")))
        log_path += f"_repflag_{rep_len}"
    return log_path


def code_snapshot(remark):
    r"""
    save code latest
    :param remark: the experiment name
    :return:
    """
    import sys
    if sys.platform.startswith("win"):
        return "win platform"
    collect_files = ["*.py", "utils/", "models/"]

    code_dir = Path(osp.join("code_snapshots", remark))
    create_dir(code_dir)
    print("code snapshot dir:", code_dir)

    for stir in collect_files:
        for filename in glob.glob(stir):
            if os.path.isdir(filename):
                subprocess.call(["cp", "-rf", filename, code_dir])
            else:
                shutil.copyfile(filename, code_dir / filename)
    return code_dir


def model_save(remark, model, epoch, is_best, fn=None, **kwargs):
    r"""
    save model to disk
    :param fn: save checkpoint name
    :param remark: experiment name
    :param model:
    :param epoch:
    :param is_best:
    :param kwargs: the duck type of some params e.g. a=1, b=1 => save_dict.update({a:1,b:1})
    """
    if fn is None:
        fn = 'now_model.pt'
    save_dir = Path(osp.join("checkpoints", remark))
    create_dir(save_dir)

    save_dict = {"state_dict": model.state_dict(), "epoch": epoch}  #存的是模型参数
    save_dict.update(kwargs) #添加best_auc和best_fitb
    torch.save(save_dict, save_dir / fn)
    best_path = save_dir / 'best_model.pt'
    if is_best:
        shutil.copyfile(save_dir / fn, best_path)   #如果是的话，将一个文件复制到另一个文件夹中

    return best_path


def model_size(model):
    r"""
    calculate the params size of model
    :param model:
    :return: model size of MB
    """
    n_parameters = sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])
    return n_parameters * 4 / 1024 / 1024


class Timer(object):
    _cid = {}

    @classmethod
    def start(cls, name):
        if name not in cls._cid:
            cls._cid[name] = time.time()
        else:
            raise InterruptedError(f"{name} is running...")

    @classmethod
    def end(cls, name):
        if name not in cls._cid:
            raise InterruptedError(f"{name} not define! All timer:{list(cls._cid.keys())}")
        return print(f"Timer[{name}]: {cls.end_time(name):.2f}s")

    @classmethod
    def end_time(cls, name):
        elapse = time.time() - cls._cid.get(name)
        del cls._cid[name]
        return elapse


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
