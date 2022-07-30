from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm
from torch.nn.init import xavier_uniform_
from torch_scatter import scatter_sum


class AttTransformerEncoderLayer(TransformerEncoderLayer):

    @staticmethod
    def _calc_scatter_index(padding_mask):
        # input: N,S  dtype=bool
        _padding_mask = padding_mask.clone().long()
        bs = _padding_mask.size(0)
        for i in range(_padding_mask.size(0)):
            _padding_mask[i][padding_mask[i]] = bs
            _padding_mask[i][padding_mask[i] == False] = i
        return _padding_mask.reshape(-1)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        src2, weight_att = self.self_attn(src, src, src, attn_mask=src_mask,
                                          key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # weight_att (N, S, S)
        index = self._calc_scatter_index(src_key_padding_mask).to(src.device)
        weight_att = scatter_sum(weight_att.view(-1, weight_att.size(-1)), index, dim=0)
        # return src, weight_att.sum(1)  # (S,N,E), (N,S)
        ret = weight_att[:-1] if src_key_padding_mask.sum().item() else weight_att
        return src, ret  # (S,N,E), (N,S)


class AttTransformerEncoder(TransformerEncoder):
    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        output = src

        layers_weight_att = []
        for mod in self.layers:
            output, weight_att = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            layers_weight_att.append(weight_att)

        if self.norm is not None:
            output = self.norm(output)

        final_weight_att = torch.stack(layers_weight_att, dim=1).sum(1)
        final_att = F.softmax(final_weight_att.masked_fill(src_key_padding_mask, float('-inf')), dim=-1)

        return output, final_att  # (S,N,E), (N,S)


class CTransformerEncoder(nn.Module):
    def __init__(self, num_node_features, nhead, dim_feedforward, num_layers, dropout=0.5):
        super(CTransformerEncoder, self).__init__()
        encoder_layer = AttTransformerEncoderLayer(num_node_features, nhead, dim_feedforward, dropout)
        encoder_norm = LayerNorm(num_node_features)
        self.encoder = AttTransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, padding_mask=None, attn_mask=None):
        return self.encoder(src, src_key_padding_mask=padding_mask, mask=attn_mask)
