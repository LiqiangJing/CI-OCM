import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch import Tensor


class GNN(torch_geometric.nn.MessagePassing):   #进行gnn的迭代
    r"""
    aggregate: like-wise graph-sage
    """

    def __init__(self, num_node_features: int, normalize: bool = False,
                 bias: bool = True, penalty_l1: bool = False, **kwargs):
        super(GNN, self).__init__(aggr='add', **kwargs)

        self.num_node_features = num_node_features
        self.normalize = normalize
        self.penalty_l1 = penalty_l1
        self.embedding_l1 = None   #字面意思，embedding的l1正则化项

        self.linear = nn.Linear(num_node_features, num_node_features, bias=bias)    #线性全连接层
        self.gate = nn.Sequential(nn.Linear(num_node_features * 2, num_node_features, bias=bias), nn.LeakyReLU(),
                                  nn.Linear(num_node_features, num_node_features, bias=bias),
                                  nn.Sigmoid())  #

        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, data) -> Tensor:
        r"""

        :param data: Batch data
        :return: GNN result
        """

        out = self.propagate(data.edge_index, x=data.x, edge_weight=data.edge_weight)   #去看看这三项都是什么
        assert out.size() == data.x.size()

        out = self.linear(data.x + out)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_weight: Tensor):  #这里传入xi和edge weight

        gate_dim = self.gate(torch.cat([x_i, x_j], dim=-1))
        # if mode==1:
        #
        # else if mode==2:

        if self.penalty_l1:
            if x_i.size(0) > 0:
                self.embedding_l1 = gate_dim.norm(p=1) / x_i.size(0)
            else:
                self.embedding_l1 = torch.zeros(1).to(x_i.device)

        return F.leaky_relu(x_j * x_i * gate_dim) * edge_weight

    def __repr__(self):
        return '{}(num_node_feature:{})'.format(self.__class__.__name__, self.num_node_features)
