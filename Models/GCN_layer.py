import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric import utils as utils


class GCN(MessagePassing):
    def __init__(self, in_size, out_size, aggr='add', flow='source_to_target'):
        super(GCN, self).__init__(aggr=aggr, flow=flow)
        self.lin = nn.Linear(in_size, out_size, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_size))

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    # x 输入形状为(N, in_size), edge_index 为(2, E)
    def forward(self, x, edge_index):
        edge_index, _ = utils.add_self_loops(edge_index, )
        x = self.lin(x)

        x_from, x_to = edge_index
        deg = utils.degree(x_from, x_from.size(0), dtype=x.dtype)
        deg = np.power(deg, -0.5)
        deg[deg == float('inf')] = 0
        norm = deg[x_from] * deg[x_to]

        out = self.propagate(edge_index, x=x, norm=norm)

        return out

    # x_j形状为[E, out_size]
    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j