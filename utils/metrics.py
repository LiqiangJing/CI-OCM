# -*- coding: utf-8 -*-
# Date: 2021/03/15 18:28

"""
metrics
"""
__author__ = 'tianyu'

import numpy as np


def MRR(array):
    """
    :param array: [bs,list], where index=0 is true
    :return:
    """
    r_sort_idx = np.argsort(-array, axis=1)
    right_idx = np.where(r_sort_idx == 0)[1]
    return np.average(1. / (right_idx + 1))


def HR(array, ks=(1, 3, 5)):
    """
    :param array:  [bs,list], where index=0 is true
    :param ks: top@K
    :return:
    """

    r_sort_idx = np.argsort(-array, axis=1)
    right_idx = np.where(r_sort_idx == 0)[1]
    HR_Ks = [right_idx < kk for kk in ks]
    return np.average(HR_Ks, axis=1)


def MRR_HR(array, HR_ks=(1, 3, 5)):
    """
    higher score, high ranking position
    :param array:  [bs,list], where index=0 is true
    """
    r_sort_idx = np.argsort(-array, axis=1)
    right_idx = np.where(r_sort_idx == 0)[1]
    HR_Ks = [right_idx < kk for kk in HR_ks]
    MRR = np.average(1. / (right_idx + 1))
    HR = np.average(HR_Ks, axis=1)
    return MRR, HR


def hr_dx(arrar, ks=(1, 2)):
    res = []
    for k in ks:
        aaaa = []
        for i in arrar:
            aa = i - i[0]
            aa[aa > 0] = 1
            aa[aa < 0] = 0
            aaa = np.sum(aa)
            if aaa + 1 <= k:
                aaaa.append(1)
            else:
                aaaa.append(0)
        res.append(np.average(np.array(aaaa)))
    return res

# if __name__ == '__main__':
#     cc = np.random.rand(1000, 1000) + 800
#     # cc = np.array([[1, 10, 9, 7, 6], [756,5,234, 236, 7], [43, 8088, 12,7, 9]])
#     print(MRR_HR(cc, HR_ks=(1, 2)))
#     print(hr_dx(cc))
