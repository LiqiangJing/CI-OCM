# -*- coding: utf-8 -*-
# Date: 2020/10/20 14:13

"""
dataset from: https://github.com/mvasil/fashion-compatibility
"""
__author__ = 'tianyu'

import json
import os.path as osp
import pickle
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torchvision import transforms
from tqdm import tqdm

from utils.tools import Timer
import matplotlib.pyplot as plt

def image_loader(path):
    return Image.open(path).convert('RGB')


def load_compatibility_questions(fn, id2im):   #直接根据compatibility文件构建训练集
    """ Returns the list of compatibility questions for the split
        [([items],label)]
        """
    with open(fn, 'r') as f:
        lines = f.readlines()


    compatibility_questions = []
    for line in lines:
        data = line.strip().split()
        compat_question = [id2im[iid] for iid in data[1:]]
        compatibility_questions.append((compat_question, int(data[0])))

    return compatibility_questions


def load_fitb_questions(fn, id2im):  #根据fill in the blank构建训练集（正确的选项构成一组为1，错误的构成一组为0）
    """ Returns the list of fill in the blank questions for the split
         [[P,N,N,N], [P,N,N,N] ]
         P:(items,label)
         N:(items,label)
         """
    with open(fn, 'r') as f:
        data = json.load(f)

    questions = []
    for item in data:
        question = item['question']
        question_items = [id2im[iid] for iid in question]
        right_id = f"{question[0].rsplit('_', maxsplit=1)[0]}_{item['blank_position']}"

        PNNN = [(question_items.copy() + [id2im[right_id]], True)]
        answer = item['answers']
        for ans in answer:
            if ans == right_id:
                continue
            PNNN.append((question_items.copy() + [id2im[ans]], False))

        questions.append(PNNN)

    return questions


def load_retrieval_questions(fn, id2im):
    """ Returns the list of fill in the blank questions for the split
         [[P,N,N,N], [P,N,N,N] ]
         P:(items,label)
         N:(items,label)
         """
    with open(fn, 'r') as f:
        data = json.load(f)

    questions = []
    print('extract from disk ...')
    for item in tqdm(data):
        question = item['question']
        question_items = [id2im[iid] for iid in question]

        PNNN = [(question_items.copy() + [id2im[item['right']]], True)]
        answer = item['candidate']
        for ans in answer:
            PNNN.append((question_items.copy() + [id2im[ans]], False))

        questions.append(PNNN)

    return questions


def load_typespaces(rootdir): #？？？？
    """ loads a mapping of pairs of types to the embedding used to
        compare them
    """
    typespace_fn = osp.join(rootdir, 'typespaces.p')
    with open(typespace_fn, 'rb') as fp:
        typespaces = pickle.load(fp)
    ts = {}
    for index, t in enumerate(typespaces):
        ts[t] = index

    return ts


class BenchmarkDataset(Dataset):
    # class vars
    _root_dir = None
    _image_dir = None
    _seman2dense = None
    _cate2dense = None
    _meta_data = None
    _class_init_flag = False
    _max_outfit_len = -1
    _call_next_epoch = 0

    __img_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    _train_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        __img_normalize,
    ])
    _inference_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        __img_normalize,
    ])

    @classmethod
    def init(cls, args):
        cls._class_init_flag = True
        cls._args = args
        cls._root_dir = osp.join(cls._args.data_dir, 'polyvore_outfits', cls._args.polyvore_split)
        cls._image_dir = osp.join(cls._args.data_dir, 'polyvore_outfits', 'images')

        with open(osp.join(cls._args.data_dir, 'polyvore_outfits', 'polyvore_item_metadata.json'), 'r') as fp:
            cls._meta_data = json.load(fp)

        with open(osp.join(cls._args.data_dir, 'polyvore_outfits', 'preprocessing.pkl'), 'rb') as fp:
            preprocessing = pickle.load(fp)
        cls._cate2dense = preprocessing.get("cate2dense")  #category是细粒度类别
        cls._seman2dense = preprocessing.get("seman2dense")  #粗粒度类别到dense  seman是粗粒度类别，将其映射到对应标签

        cls._co_type_weight = cls._calc_co_weight(cls._seman2dense, 'semantic_category')   #使用的是粗粒度的类别  这里改成细粒度类别是不是也可以
        #cls._co_type_weight = cls._calc_co_weight(cls._cate2dense, "category_id")


    @property
    def num_category(self):
        return len(self._cate2dense)

    @property
    def num_semantic_category(self):
        return len(self._seman2dense)

    @property
    def num_cotpye(self):
        return len(self.typespaces)

    @classmethod
    def _get_im_dense_type(cls, im, which_dense, dense_key):   #统计某一属性出现次数
        r"""
        :param im: imamge id
        :param which_dense:
        :return:
        """
        cate = cls._meta_data[im][dense_key]
        return which_dense[cate]

    @classmethod
    def _calc_co_weight(cls, dense_mapping, dense_key):#
        r"""
        the weight of the static graph by data-driven manner.
        """
        data_json = osp.join(cls._root_dir, 'train.json')  #原来是train.json
        with open(data_json, 'r') as fp:
            outfit_data = json.load(fp)

        num_category = len(dense_mapping)
        total_graph = np.zeros((num_category, num_category), dtype=np.float32)

        # count co-concurrence times
        for outfit in outfit_data:   #cate_list是从一个outfit中抽出得到的
            cate_list = outfit['items']
            cls._max_outfit_len = max(cls._max_outfit_len, len(cate_list))
            for i in range(len(cate_list)):
                rcid = cls._get_im_dense_type(cate_list[i]["item_id"], dense_mapping, dense_key)  #输入是id，dense_mapping.dense_key
                for j in range(i + 1, len(cate_list)):
                    rcjd = cls._get_im_dense_type(cate_list[j]["item_id"], dense_mapping, dense_key)
                    total_graph[rcid][rcjd] += 1.
                    total_graph[rcjd][rcid] += 1.

        total_graph /= total_graph.sum(0)    #对邻接矩阵进行归一化
        total_graph /= total_graph.sum(1, keepdims=True)
        print(cls._max_outfit_len)
        return total_graph   #total_graph是numpy数组类型

    def __init__(self, split):
        assert self._class_init_flag, f"Init:{self._class_init_flag}-> " \
                                      f"you must init class firstly by calling BenchmarkDataset.init(args)"

        data_json = osp.join(self._root_dir, '%s.json' % split)
        with open(data_json, 'r') as fp:
            outfit_data = json.load(fp)   #train.json

        # get list of images and make a mapping used to quickly organize the data
        im2type = {}
        category2ims = {}
        imnames = set()
        id2im = {}
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            for item in outfit['items']:
                im = item['item_id']   #im_id
                seman_category = self._meta_data[im]['semantic_category']
                cate_category = self._meta_data[im]['category_id']   #使用这个得到category_id
                im2type[im] = (seman_category, cate_category)

                category2ims.setdefault(seman_category, [])
                category2ims[seman_category].append(im)   #得到所有该seman_category下的id

                id2im['%s_%i' % (outfit_id, item['index'])] = im   #根据outfit set和索引得到item_id
                imnames.add(im)

        imnames = list(imnames)
        self.imnames = imnames
        self.im2type = im2type
        self.id2im = id2im
        self.category2ims = category2ims
        self.split = split
        self.typespaces = load_typespaces(self._root_dir)

        self.num_negative = None
        if self.split == 'train':
            self.pos_list = self._collect_pos_sample(outfit_data)
            self.neg_list = None
        else:
            self.kpi_list = None

    def next_train_epoch(self, num_negative=1):    #获得下一个train batch
        self._call_next_epoch += 1

        begin_same, begin_one = 10, 40
        if self._call_next_epoch < begin_same:
            print('+++ rand sample ++')
        elif self._call_next_epoch < begin_one:
            print('+++ same type sample ++')
        else:
            print('+++ replace one sample ++')
        self.num_negative = num_negative
        Timer.start('train_neg')
        # negative sample strategy: random -> same type -> replace
        self.neg_list = self._generate_neg_sample_rand2same2one(begin_same, begin_one)
        Timer.end('train_neg')

        return self

    def test_auc(self, file_name_format="compatibility_%s.txt"):
        ret = []
        fn = osp.join(self._root_dir, file_name_format % self.split)
        compatibility_questions = load_compatibility_questions(fn, self.id2im)

        for items, label in compatibility_questions:
            ret_data = self._wrapper(items, label)
            ret.append([ret_data])

        self.kpi_list = ret
        return self

    def test_fitb(self, file_name_format='fill_in_blank_%s.json'):
        ret = []
        fn = osp.join(self._root_dir, file_name_format % self.split)
        fitb_questions = load_fitb_questions(fn, self.id2im)
        for fitb in fitb_questions:
            # fitb: [P,N,N,N]
            post_fitb = []
            for each in fitb:
                # each: (items, label)
                ret_data = self._wrapper(*each)
                post_fitb.append(ret_data)
            # post_fitb: [P,N,N,N]
            ret.append(post_fitb)

        self.kpi_list = ret
        return self

    def test_retrieval(self):
        ret = []
        fn = osp.join(self._args.data_dir, 'polyvore_outfits', 'retrieval', "{}.json".format(self._args.polyvore_split))
        retrieval_questions = load_retrieval_questions(fn, self.id2im)

        print('pack to kpi list ...')
        for fitb in tqdm(retrieval_questions):
            # fitb: [P,N,N,N]
            post_fitb = []
            for each in fitb:
                # each: (items, label)
                ret_data = self._wrapper(*each)
                post_fitb.append(ret_data)
            # post_fitb: [P,N,N,N]
            ret.append(post_fitb)

        self.kpi_list = ret
        return self

    def _collect_pos_sample(self, outfit_data):
        ret = []
        for outfit in outfit_data:#
            items = list(map(lambda dic: dic["item_id"], outfit['items']))
            ret_data = self._wrapper(items, True)
            ret.append(ret_data)

        return ret
    #_wrapper每次读入一个outfit
    def _wrapper(self, file_items, compatibility):  #将该item下对应的各属性封装成一个data类
        index_source, index_target, edge_weight, type_embedding, y = [], [], [], [], [int(compatibility)]
        rcid_index = []
        scid_index = []
        for j, j_item in enumerate(file_items):
            sema_raw_j, cate_raw_j = self.im2type[j_item]
            s_j_rcid, j_rcid = self._seman2dense[sema_raw_j], self._cate2dense[cate_raw_j]#这里能否留下？？？？
            rcid_index.append(j_rcid)
            scid_index.append(s_j_rcid)
            for i, i_item in enumerate(file_items):
                if i == j:
                    continue
                index_source.append(j)
                index_target.append(i)  #增加一条有向边，之后可以通过统计有向边的个数计算边权

                sema_raw_i, cate_raw_i = self.im2type[i_item]
                s_i_rcid, i_rcid = self._seman2dense[sema_raw_i], self._cate2dense[cate_raw_i]
                #edge_weight 为list类型
                edge_weight.append(self._co_type_weight[s_j_rcid][s_i_rcid])  # probs of j, given i   修改这里可以得到细粒度类别图
                type_embedding.append(self._get_typespace(sema_raw_j, sema_raw_i))  #得到两两一对的点之间是否有边的组合
        # print(rcid_index)
        data = Data(rcid_index=torch.tensor(rcid_index, dtype=torch.long),
                    scid_index=torch.tensor(scid_index, dtype=torch.long),
                    file_items=file_items,
                    edge_index=torch.tensor([index_source, index_target], dtype=torch.long),
                    edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
                    type_embedding=torch.tensor(type_embedding, dtype=torch.long),
                    y=torch.tensor(y).float())
        return data

    def _generate_neg_sample_rand2same2one(self, begin_same, begin_one):#从这里确定变换采样方式的节点，并更改到调用该函数的地方
        ret = []
        for i, pos in enumerate(self.pos_list):
            if self._call_next_epoch < begin_same:
                ret.append(self._neg_rand(i, len(pos["file_items"]), self.num_negative))
            elif self._call_next_epoch < begin_one:
                ret.append(self._neg_rand_same_type(i, pos["file_items"], self.num_negative))
            else:
                ret.append(self._neg_rand_same_type_one(i, pos["file_items"], self.num_negative))

        print('negative sample done!')
        return ret

    def _neg_rand(self, i, pos_len, num_negative):
        if i and i % 5000 == 0:
            print(f"neg sample at {i}")
        neg_outfits = []
        neg_i = 0

        tot_len = len(self.imnames)
        while neg_i < num_negative:
            neg_len = pos_len
            neg_outfit_ids = []
            for i in range(neg_len):
                while True:
                    nno = np.random.randint(0, tot_len)
                    neg = self.imnames[nno]
                    if neg not in neg_outfit_ids:  # no the same item in one outfit
                        break

                neg_outfit_ids.append(neg)
            # construct Data
            neg_data = self._wrapper(neg_outfit_ids, False)
            neg_outfits.append(neg_data)

            neg_i += 1
        return neg_outfits

    def _neg_rand_same_type(self, i, file_items, num_negative):
        if i and i % 5000 == 0:
            print(f"neg sample at {i}")
        neg_outfits = []
        neg_i = 0

        while neg_i < num_negative:
            neg_len = len(file_items)
            neg_outfit_ids = []
            for i in range(neg_len):
                while True:
                    sem, _ = self.im2type[file_items[i]]
                    candidate_sets = self.category2ims[sem]
                    nno = np.random.randint(0, len(candidate_sets))
                    neg = candidate_sets[nno]

                    if neg not in neg_outfit_ids:  # no the same item in one outfit
                        break

                neg_outfit_ids.append(neg)
            # construct Data
            neg_data = self._wrapper(neg_outfit_ids, False)
            neg_outfits.append(neg_data)

            neg_i += 1
        return neg_outfits

    def _neg_rand_same_type_one(self, i, file_items, num_negative):

        def com_sample():
            tmp_neg_iid = []
            tmp_num_neg = num_negative
            if neg_len < num_negative:
                while neg_len < tmp_num_neg:
                    tmp_neg_iid += random.sample(range(neg_len), k=neg_len)
                    tmp_num_neg -= neg_len

            return tmp_neg_iid + random.sample(range(neg_len), k=tmp_num_neg)

        if i and i % 5000 == 0:
            print(f"neg sample at {i}")
        neg_outfits = []
        neg_i = 0

        neg_len = len(file_items)
        neg_replace_pos = com_sample()
        while neg_i < num_negative:
            neg_outfit_ids = file_items.copy()
            while True:
                sem, _ = self.im2type[file_items[neg_replace_pos[neg_i]]]
                candidate_sets = self.category2ims[sem]
                nno = np.random.randint(0, len(candidate_sets))
                neg = candidate_sets[nno]

                if neg not in neg_outfit_ids:  # no the same item in one outfit
                    break

            neg_outfit_ids[neg_replace_pos[neg_i]] = neg

            # construct Data
            neg_data = self._wrapper(neg_outfit_ids, False)
            neg_outfits.append(neg_data)

            neg_i += 1
        return neg_outfits

    def _get_typespace(self, anchor, pair):
        """ Returns the index of the type specific embedding
            for the pair of item types provided as input
        """
        query = (anchor, pair)
        if query not in self.typespaces:
            query = (pair, anchor)

        return self.typespaces[query]

    def _fetch_img(self, img_fns):

        if self.split == 'train':
            apply_transform = self._train_transform
        else:
            apply_transform = self._inference_transform

        img_data = []
        for fn in img_fns:
            img = image_loader(osp.join(self._image_dir, fn))
            img_data.append(apply_transform(img))

        return torch.stack(img_data, dim=0)  # N,3,112,112

    def __getitem__(self, index):
        if self.split == 'train':
            bundle = [self.pos_list[index].clone()] + [obj.clone() for obj in self.neg_list[index]]
        else:
            bundle = [obj.clone() for obj in self.kpi_list[index]]

        for one in bundle:
            img_fns = [f"{iid}.jpg" for iid in one["file_items"]]  #这个是从哪里来的
            # print(type(img_fns))
            #你怎么都没发现根本就没有转换过来！！！！！！！
            one.c=[self._cate2dense[self._meta_data[im]['category_id']] for im in one["file_items"]]  #这里是否需要改进一下，得到关于类别的双重list
            one.x = self._fetch_img(img_fns)   #x中保存的是对应的图片数据
        # print(bundle)
        return bundle

    def __len__(self):
        if self.split == 'train':
            return len(self.pos_list)

        return len(self.kpi_list)
