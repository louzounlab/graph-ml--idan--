import torch
import torch.nn as nn
import torch.nn.functional as functional
from common.layers import GraphConvolution, AsymmetricGCN


class GCNKipf(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNKipf, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self._activation_func = functional.relu
        self._dropout = dropout

    def forward(self, x, adj):
        x = self._activation_func(self.gc1(x, adj))
        x = functional.dropout(x, self._dropout, training=self.training)
        x = self.gc2(x, adj)
        return functional.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, nfeat, hlayers, nclass, dropout, last_layer=True, is_asym=False):
        super(GCN, self).__init__()
        layer_type = AsymmetricGCN if is_asym else GraphConvolution

        hlayers = [nfeat] + hlayers
        self.layers = nn.ModuleList([layer_type(first, second) for first, second in zip(hlayers[:-1], hlayers[1:])])
        self.class_layer = layer_type(hlayers[-1], nclass)

        self._activation_func = functional.relu
        self._dropout = dropout
        self._is_asym = is_asym
        self.last_layer = last_layer

    @property
    def n_output(self):
        if self.last_layer:
            return self.class_layer.out_features
        if self.layers:
            return self.layers[-1].out_features

    def forward(self, x, adj, dropout=True):
        for layer in self.layers:
            x = self._activation_func(layer(x, adj))
            if dropout:
                x = functional.dropout(x, self._dropout, training=self.training)

        if self.last_layer:
            x = self.class_layer(x, adj)
            return functional.log_softmax(x, dim=1)
        return x


class GCNCombined(GCN):
    def __init__(self, nbow, nfeat, hlayers, nclass, dropout, last_layer=True, is_asym=True):
        num_input = hlayers[0]
        if is_asym:
            num_input *= 2
        num_input += nfeat
        super(GCNCombined, self).__init__(num_input, hlayers[1:], nclass, dropout,
                                          last_layer=last_layer, is_asym=is_asym)
        self.bow_layer = GraphConvolution(nbow, hlayers[0])

    # noinspection PyMethodOverriding
    def forward(self, bow, feat, adj):
        x = self._activation_func(self.bow_layer(bow, adj))
        x = functional.dropout(x, self._dropout, training=self.training)
        if self._is_asym:
            x = torch.cat(torch.chunk(x, 2, dim=0), dim=1)
        x = torch.cat([x, feat], dim=1)

        return super(GCNCombined, self).forward(x, adj, dropout=False)


# super(GCNCombined, self).__init__()
# self._is_asym = is_asym
# hlayers = hlayers[:]
# self.bow_layer = GraphConvolution(nbow, hlayers[0])
# layer_type = GraphConvolution
# if self._is_asym:
#     layer_type = AsymmetricGCN
#     hlayers[0] *= 2
#
# hlayers[0] += nfeat
# self.layers = nn.ModuleList([layer_type(first, second) for first, second in zip(hlayers[:-1], hlayers[1:])])
# self.class_layer = layer_type(hlayers[-1], nclass)
#
# self._activation_func = functional.relu
# self._dropout = dropout
# self.last_layer = last_layer
