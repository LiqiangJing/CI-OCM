# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 22:18:42 2022

@author: DELL
"""
import pickle
import os.path as osp

with open('preprocessing.pkl', 'rb') as fp:
    preprocessing = pickle.load(fp)

print(type(preprocessing)) #dict类型
print(preprocessing)
print(type(preprocessing['cate2dense']['15']))