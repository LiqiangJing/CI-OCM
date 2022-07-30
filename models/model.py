import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os.path as osp

from .branch_module import GABlock
from .resnet_18 import resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BranchScore(nn.Module):
    def __init__(self, **kwargs):
        super(BranchScore, self).__init__()
        self.trans_w = nn.Linear(kwargs["node_dim"], kwargs["node_dim"], bias=False)
        self.attr_gablock = GABlock(**kwargs)   #branch_module文件中
        self.attr_score = nn.Linear(self.attr_gablock.output_dim, 1)

        self.reset_parameters()   #重置参数

    @staticmethod
    def _init_sequential(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.001)
            nn.init.constant_(m.bias, 0)

    def reset_parameters(self):
        self.attr_score.apply(self._init_sequential)

    def forward(self, data):
        img = data.x
        img2attr = self.trans_w(img)
        data.x = img2attr
        attr, type_mask_norm = self.attr_gablock(data)  # (Graph_N,D)   #gather（P），l1正则化项
        score_attr = self.attr_score(attr)
        return img2attr, attr, score_attr, type_mask_norm


class CORL(nn.Module):
    def __init__(self, attr_aspect, **kwargs):
        super(CORL, self).__init__()
        self.attr_aspect = attr_aspect   #方向数
        self.attr_branches = self._make_branch(attr_aspect, **kwargs)  #每个方向分支对应的模型

    @staticmethod
    def _make_branch(attr_aspect, **kwargs):
        branches = nn.ModuleList()
        for branch in range(attr_aspect):
            branches.append(BranchScore(**kwargs))    #定义在上方
        return branches

    def forward(self, data):
        img_node = data.x.clone()  # N,D
        img2attrs, attrs, score_attrs, type_mask_norms = [], [], [], []
        for branch in self.attr_branches:
            data.x = img_node
            img2attr, attr, score_attr, type_mask_norm = branch(data)  # (N,D),(Graph_N,D),(N,1),(1)
            img2attrs.append(img2attr)
            attrs.append(attr)
            score_attrs.append(score_attr)
            type_mask_norms.append(type_mask_norm)

        # complementarity regularization loss
        mat_feats = F.normalize(torch.stack(img2attrs, dim=1), p=2, dim=-1)  # N,branch,D
        diversity_mat = torch.bmm(mat_feats, mat_feats.transpose(1, 2))  # N,branch,branch
        eye_mat = torch.eye(diversity_mat.size(-1)).unsqueeze(0).repeat(diversity_mat.size(0), 1, 1).to(
            diversity_mat.device)
        diversity_loss = torch.pow(eye_mat - diversity_mat, exponent=2).sum()   #l2正则化

        # type mask norm
        tot_type_mask_norm = torch.stack(type_mask_norms).mean()  #l1正则化项

        # fine score
        fine_score = torch.stack(score_attrs, dim=1).sum(dim=1)


        attr_weight = F.softmax(torch.cat(score_attrs, dim=-1), dim=-1)  # N,branch
        fine_feature = torch.bmm(attr_weight.unsqueeze(1), torch.stack(attrs, dim=1)).squeeze(1)  # N,D   特征
        return fine_feature, fine_score, tot_type_mask_norm, diversity_loss


class CF(nn.Module):
    def __init__(self, num_node_features: int, num_cotpye: int, depth: int, nhead: int,
                 dim_feedforward: int, num_layers: int, num_category: int, num_branch: int = 5):
        super(CF, self).__init__()
        # self.cotpye_embedding = nn.Embedding(num_cotpye + 1, num_node_features)  # 加个 L1 loss 需要稀疏
        self.num_cotpye = num_cotpye
        self.num_node_features = num_node_features
        self.depth = depth
        self.nhead = nhead
        self.num_category = num_category
        self.num_negative = 1

        self.embedding = resnet18(pretrained=True, embedding_size=self.num_node_features)

        self.disentangle_gablock = CORL(attr_aspect=num_branch, node_dim=num_node_features, gnn_deep=depth,
                                        attention_head=nhead, attention_layer=num_layers,
                                        attention_feed_dim=dim_feedforward)

    #寻找从哪里分叉比较合适
    def forward(self, data):
        # get img embed
        data.x = self.embedding(data.x)   #视觉信息
        img_embed_norm = data.x.norm(2) / np.sqrt(data.x.size(0))#将得到的embedding归一化

        disentangle_data = data.clone()

        # attr branch
        fine_feature, fine_score, attr_type_mask_norm, diversity_loss = self.disentangle_gablock(disentangle_data)

        return fine_score, attr_type_mask_norm, img_embed_norm, diversity_loss   #score,l1,

    def bpr_loss(self, output): #bayes损失函数
        output = output.view(-1, (self.num_negative + 1))  # each row: (pos, neg, neg, neg, ..., neg)  view(-1):将原张量变成一维的张量

        # the first score (pos scores) minus each remainder scores (neg scores)
        output = output[:, 0].unsqueeze(-1).expand_as(output[:, 1:]) - output[:, 1:]   #将positive的输出复制多个并分别减去一个negtive的值

        batch_acc = (output > 0).sum().item() * 1.0 / output.nelement()   #batch的正确率（正确的分对的-错误的分成错的）  计算分到正类比分到负类概率大的item，计算其一共有多少个，从而得到正确率

        return -F.logsigmoid(output).mean(), batch_acc

    def bce_loss(self,output,y):
        m = nn.Sigmoid()
        criterion = nn.BCELoss()

        y=y.unsqueeze(1)
        y=y.float()
        loss = criterion(m(output), y)
        output = output.view(-1, (self.num_negative + 1))  # each row: (pos, neg, neg, neg, ..., neg)  view(-1):将原张量变成一维的张量

        # the first score (pos scores) minus each remainder scores (neg scores)
        output = output[:, 0].unsqueeze(-1).expand_as(output[:, 1:]) - output[:, 1:]   #将positive的输出复制多个并分别减去一个negtive的值

        batch_acc = (output > 0).sum().item() * 1.0 / output.nelement()   #batch的正确率（正确的分对的-错误的分成错的）  计算分到正类比分到负类概率大的item，计算其一共有多少个，从而得到正确率

        return loss, batch_acc
    @torch.no_grad()   #下面的数据不需要计算梯度，也不需要反向传播
    def test_fitb(self, batch):
        self.eval()
        output, _, _,_= self(batch)
        output = output.view(-1, 4)  # each row: (pos, neg, neg, neg)
        _, max_idx = output.max(dim=-1)
        return (max_idx == 0).sum().item()

    @torch.no_grad()
    def test_auc(self, batch):
        self.eval()
        output, _, _,_ = self(batch)
        return output.view(-1)

    @torch.no_grad()
    def test_retrieval(self, batch, ranking_neg_num):
        self.eval()
        output = self(batch)[0]
        return output.view(-1, ranking_neg_num + 1)

class CF2(nn.Module):
    def __init__(self, num_node_features: int, num_cotpye: int, depth: int, nhead: int,
                 dim_feedforward: int, num_layers: int, num_category: int, num_branch: int = 5,):
        super(CF2, self).__init__()
        self.num_cotpye = num_cotpye
        self.num_node_features = num_node_features
        self.depth = depth
        self.nhead = nhead
        self.num_category = num_category
        self.num_negative = 1
        self.input_size=155
        self.embed_size=10


        self.embed = nn.Embedding(self.input_size, self.embed_size, padding_idx=153)


        # self.cotpye_embedding = nn.Embedding(num_cotpye + 1, num_node_features)  # 加个 L1 loss 需要稀疏
        self.category=nn.ModuleList(nn.Sequential(
            nn.Linear(190, 190),
            nn.ReLU()) for i in range(2))
        # self.fc1=torch.nn.Linear(190,128)
        # self.fc2=torch.nn.Linear(128,64)  #减少中间节点的个数
        self.out=nn.Linear(190,1)  #变成二分类问题
        self.category_visual=CF(num_node_features, num_cotpye,depth, nhead, dim_feedforward,num_layers, num_category)



    #寻找从哪里分叉比较合适
    def forward(self, data):
        y=data.y
        data1=data.clone()
        data2=data.clone()
        data2=data2.c

        for i in range(len(data2)):
            data2[i]+=[153 for i in range(19-len(data2[i]))]

        data2 = torch.LongTensor(data2) #取出
        #从data2中分理出类别信息，按照原类别id的二维列表形式进行组合：
        #新的text分支：

        embedded = self.embed(data2.to('cuda'))
        # print("embedded:",embedded.shape)  #128,19,10

        embedded=embedded.view(-1,19*10)


        text_detach = self.category[0](embedded)
        # # print("embedded:", text_detach.shape)
        text_detach=self.category[1](text_detach)

        ret2=self.out(text_detach)  #738,1  batchsize*outfit_size

        loss_c,_=self.bce_loss(ret2,y) #Lc

        ret1, type_mask_norm1, img_embed_norm1, diversity_norm1=self.category_visual(data1)

        loss_type_mask1 = 5e-4 * type_mask_norm1
        loss_img_embed1 = 5e-3 * img_embed_norm1
        loss_diversity1 = 5e-3 * diversity_norm1
        loss_other = loss_type_mask1 + loss_img_embed1 + loss_diversity1

        outlayer = nn.Sigmoid()
        ret2=outlayer(ret2) #sigmoid(yc)
        ret=ret1*ret2  #ycv=yk*sigmoid(yc)
        loss_o,batch_acc=self.bce_loss(ret,y)
        alpha=1e-3
        loss = loss_o + alpha * loss_c + loss_other

        return ret,ret2,loss,batch_acc #

    def bpr_loss(self, output):
        output = output.view(-1, (self.num_negative + 1))  # each row: (pos, neg, neg, neg, ..., neg)

        # the first score (pos scores) minus each remainder scores (neg scores)
        output = output[:, 0].unsqueeze(-1).expand_as(output[:, 1:]) - output[:, 1:]

        batch_acc = (output > 0).sum().item() * 1.0 / output.nelement()

        return -F.logsigmoid(output).mean(), batch_acc

    def bce_loss(self,output,y):
        m = nn.Sigmoid()
        criterion = nn.BCELoss()

        y=y.unsqueeze(1)
        y=y.float()
        loss = criterion(m(output), y)

        output = output.view(-1, (self.num_negative + 1))  # each row: (pos, neg, neg, neg, ..., neg)  view(-1):将原张量变成一维的张量

        # the first score (pos scores) minus each remainder scores (neg scores)
        output = output[:, 0].unsqueeze(-1).expand_as(output[:, 1:]) - output[:, 1:]   #将positive的输出复制多个并分别减去一个negtive的值

        batch_acc = (output > 0).sum().item() * 1.0 / output.nelement()   #batch的正确率（正确的分对的-错误的分成错的）  计算分到正类比分到负类概率大的item，计算其一共有多少个，从而得到正确率

        return loss, batch_acc

    @torch.no_grad()
    def test_fitb(self, batch,c):
        self.eval()
        ret1,ret2, _, _ = self(batch)
        output=ret1-c*ret2
        output = output.view(-1, 4)  # each row: (pos, neg, neg, neg)
        _, max_idx = output.max(dim=-1)
        return (max_idx == 0).sum().item()

    @torch.no_grad()
    def test_auc(self, batch,c):
        self.eval()
        ret1,ret2, _, _= self(batch)
        output=ret1-c*ret2
        return output.view(-1)

    @torch.no_grad()
    def test_retrieval(self, batch, ranking_neg_num,c):
        self.eval()
        ret1,ret2,_,_ = self(batch)
        output=ret1-c*ret2
        return output.view(-1, ranking_neg_num + 1)


    @torch.no_grad()
    def test_fitb2(self, batch,c):
        self.eval()
        ret1,ret2, _, _ = self(batch)
        output=ret1-c*ret2
        output2=ret2
        output = output.view(-1, 4)  # each row: (pos, neg, neg, neg)
        output2=output2.view(-1,4)
        _, max_idx = output.max(dim=-1)  #第一个是可以正确匹配的
        _,max_idx2=output2.max(dim=-1)
        return (max_idx == 0).sum().item(),(max_idx2 == 0).sum().item()

    @torch.no_grad()
    def test_auc2(self, batch,c):
        self.eval()
        ret1,ret2, _, _= self(batch)
        output=ret1-c*ret2
        output2=ret2
        return output.view(-1),output2.view(-1)

    @torch.no_grad()
    def test_retrieval2(self, batch, ranking_neg_num,c):
        self.eval()
        ret1,ret2,_,_ = self(batch)
        output=ret1-c*ret2
        output2=ret2
        return output.view(-1, ranking_neg_num + 1),output2.view(-1, ranking_neg_num + 1)
