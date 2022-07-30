import os.path as osp
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

def get_im_dense_type(meta_data,im, which_dense, dense_key):  # 统计某一属性出现次数
    r"""
    :param im: imamge id
    :param which_dense:
    :return:
    """
    cate = meta_data[im][dense_key]
    return which_dense[cate]


root_dir='nondisjoint'


with open('preprocessing.pkl', 'rb') as fp:
    preprocessing = pickle.load(fp)
#类别到索引的字典
cate2dense = preprocessing.get("cate2dense")  # category是细粒度类别
seman2dense = preprocessing.get("seman2dense")  # 粗粒度类别到dense  seman是粗粒度类别，将其映射到对应标签

dense_mapping=seman2dense
dense_key='semantic_category'


#load meta_data 数据
with open( 'polyvore_item_metadata.json', 'r') as fp:
    meta_data = json.load(fp)

#load train,valid,test 并且计算train,valid,test各自所占比例，同时合并三个集合
data_train_json = osp.join(root_dir, 'train.json')
data_valid_json = osp.join(root_dir, 'valid.json')
data_test_json = osp.join(root_dir, 'test.json')
with open(data_train_json, 'r') as fp:
    train_data = json.load(fp)     #不需要close
with open(data_valid_json, 'r') as fp:
    valid_data = json.load(fp)     #不需要close
with open(data_test_json, 'r') as fp:
    test_data = json.load(fp)     #不需要close

train_len=len(train_data)
valid_len=len(valid_data)
test_len=len(test_data)
sum_len=len(train_data)+len(valid_data)+len(test_data)
print('train数据集大小：',train_len)
print('valid数据集大小：',valid_len)
print('test数据集大小：',test_len)
print('完整数据集大小：',sum_len)
print('train所占比例',train_len/sum_len)
print('valid所占比例',valid_len/sum_len)
print('test所占比例',test_len/sum_len)
outfit_data= train_data+valid_data+test_data

print('合并后的数据集大小:',len(outfit_data))



num_category = len(dense_mapping)
total_graph = np.zeros((num_category, num_category), dtype=np.float32)
max_outfit_len=-1
for outfit in outfit_data:  # cate_list是从一个outfit中抽出得到的
    cate_list = outfit['items']
    max_outfit_len = max(max_outfit_len, len(cate_list))
    for i in range(len(cate_list)):
        rcid = get_im_dense_type(meta_data,cate_list[i]["item_id"], dense_mapping,
                                      dense_key)  # 输入是id，dense_mapping.dense_key
        for j in range(i + 1, len(cate_list)):
            rcjd = get_im_dense_type(meta_data,cate_list[j]["item_id"], dense_mapping, dense_key)
            total_graph[rcid][rcjd] += 1.
            total_graph[rcjd][rcid] += 1.

for i in range(num_category):
    cate2num={}
    for j in range(num_category):
        key=str(i)+'-'+str(j)
        cate2num[key]=total_graph[i][j]
    sort_cnt = sorted(cate2num.items(), key=lambda x: x[1], reverse=True)
    np.save(dense_key + '_' + str(i) + '.npy', sort_cnt)
    # # 先创建并打开一个文本文件
    file = open(dense_key + '_' + str(i) + '.txt', 'w')
    # 遍历字典的元素，将每项元素的key和value分拆组成字符串，注意添加分隔符和换行符
    for tmp in sort_cnt:
        file.write(str(tmp[0]) + ' ' + str(tmp[1]) + '\n')
    # 注意关闭文件
    file.close()
    # jsObj = json.dumps(sort_cnt)
    # fileObject = open(dense_key+'_'+str(i)+'.json', 'w')
    # fileObject.write(jsObj)
    # fileObject.close()

    # tf = open(dense_key+'_'+str(i)+'.json', "w")
    # json.dump(sort_cnt, tf)
    # tf.close()
    # 绘制图片：
    x = []  # 绘制图片的横纵坐标
    y = []
    for tmp in sort_cnt:
        x.append(tmp[0])
        y.append(tmp[1])
    plt.bar(x, y)
    plt.savefig(dense_key+'_'+str(i)+'.jpg')
    plt.clf()


#所有类别进行排序并输出文件
catesum2num={}
for i in range(num_category):
    for j in range(i,num_category):
        key=str(i)+'-'+str(j)
        catesum2num[key]=total_graph[i][j]
sort_cnt = sorted(catesum2num.items(), key=lambda x: x[1], reverse=True)
print(sort_cnt)
print(type(sort_cnt))
np.save(dense_key + '.npy', sort_cnt)
# # 先创建并打开一个文本文件
file = open(dense_key + '.txt', 'w')
# 遍历字典的元素，将每项元素的key和value分拆组成字符串，注意添加分隔符和换行符
for tmp in sort_cnt:
    file.write(str(tmp[0]) + ' ' + str(tmp[1]) + '\n')
# 注意关闭文件
file.close()
# jsObj = json.dumps(sort_cnt)
# fileObject = open(dense_key + '.json', 'w')
# fileObject.write(jsObj)
# fileObject.close()
#
# tf = open(dense_key + '.json', "w")
# json.dump(sort_cnt, tf)
# tf.close()
x=[]
y=[]
for tmp in sort_cnt:
    x.append(tmp[0])
    y.append(tmp[1])
plt.bar(x, y)
plt.savefig(dense_key + '.jpg')
plt.clf()