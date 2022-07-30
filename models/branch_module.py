import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .gnn_module import GNN
from .transformer_module import CTransformerEncoder

#graph attribute?
class GABlock(nn.Module):#增加mode 选择边和点的输入
    def __init__(self, node_dim, gnn_deep, attention_head, attention_layer, attention_feed_dim):
        super(GABlock, self).__init__()
        self.node_dim = node_dim
        self.gnn_deep = gnn_deep
        self.output_dim = node_dim * 2
        self.dropout = 0.0

        # build layers
        self.gnn = self._make_gnn_module()  #一层gnn一层relu一层dropout，最后一层不加dropout  gnn的迭代过程
        self.attention = CTransformerEncoder(self.output_dim, attention_head, attention_feed_dim,
                                             attention_layer, dropout=self.dropout)  #在transformer module中，

    def _make_gnn_module(self):
        gnn = nn.ModuleList()
        for i in range(self.gnn_deep):  #gnn的深度
            gnn.append(GNN(self.node_dim, bias=True, penalty_l1=True, normalize=False))
            gnn.append(nn.LeakyReLU(0.01))
            if i != self.gnn_deep - 1:
                gnn.append(nn.Dropout(p=self.dropout))
        return gnn

    @staticmethod
    def _calc_src_key_padding_mask(graphs, is_bool=True):
        max_len = max([s.size(0) for s in graphs])
        padding_mask = torch.ones(len(graphs), max_len)
        for i, graph in enumerate(graphs):
            index = torch.tensor([max_len - ti for ti in range(1, max_len - graph.size(0) + 1)])
            if len(index):
                padding_mask[i].index_fill_(0, index, 0)
        if is_bool:
            return (1 - padding_mask).bool()
        else:
            return padding_mask

    def forward(self, data):
        r"""
        preprocessed data
        :param data: type of `torch_geometric.data.Data`
        :return:  pooled node of a graph, loss
        """
        before_gnn = data.x.clone()
        gnn_out = data
        gnn_type_mask_norm_container = []
        for idx, gnn in enumerate(self.gnn):
            if idx and isinstance(gnn, GNN):  #遍历gnn modulelist中的所有项
                data.x = gnn_out
                gnn_out = gnn(data)
            else:
                gnn_out = gnn(gnn_out)
            if isinstance(gnn, GNN):
                gnn_type_mask_norm_container.append(gnn.embedding_l1)   #l1正则化项

        type_mask_norm = torch.stack(gnn_type_mask_norm_container).mean()#沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。

        gnn_out = torch.cat([before_gnn, gnn_out], dim=-1)  #原先的和gnn合并在一起

        # convert data structure for attention layer
        graphs = []
        for shift in range(data.num_graphs):
            tmp_slices = data.slices_indicator[shift:shift + 2]
            graphs.append(gnn_out[tmp_slices[0].item():tmp_slices[1].item()])
        batch_data = pad_sequence(graphs)  # seq,batchsize,hidden
        padding_mask = self._calc_src_key_padding_mask(graphs).to(gnn_out.device)

        output, final_att = self.attention(batch_data, padding_mask)  # (S,N,E), (N,S)
        # print(final_att)
        output.transpose_(0, 1)  # (N,S,E)
        # noinspection PyTypeChecker
        output = torch.einsum('bij,bi->bij', [output, final_att])  # (N,S,E)  sum
        gather = output.sum(1)  # (N,E)

        return gather, type_mask_norm
