from torch.nn import ModuleList
from torch_geometric import nn as pnn
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool
from torch_geometric.utils import degree

import config
from Models.Loss.MV_GNN_MSE import *
from config import *


def get_deg(train_dataset):
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg


class PNA_Net(nn.Module):
    def __init__(self,
                 node_channels,
                 edge_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 num_classes,
                 drop_p,
                 deg):
        super().__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        for _ in range(num_layers):
            conv = PNAConv(in_channels=node_channels, out_channels=node_channels,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=edge_channels, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(node_channels))
        self.lin1 = pnn.Linear(in_channels=node_channels,
                               out_channels=hidden_channels)
        self.lin2 = pnn.Linear(in_channels=hidden_channels,
                               out_channels=out_channels)
        self.linout = pnn.Linear(in_channels=out_channels,
                                 out_channels=num_classes)

        self.device = config.device

    def forward(self, X):
        edge_index = X.edge_index
        node_feature = X.x
        edge_feature = X.edge_attr
        edge_index = edge_index.to(self.device)
        node_feature = node_feature.to(self.device)
        batch = X.batch.to(self.device)
        edge_feature = edge_feature.to(self.device)

        # def forward(self, x: Tensor, edge_index: Adj,
        #             edge_attr: OptTensor = None) -> Tensor:
        hid = node_feature
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            hid = torch.relu(batch_norm(conv(hid, edge_index, edge_feature)))

        out = self.lin1(hid)
        out = torch.relu(out)
        out = self.lin2(out)
        out = torch.relu(out)
        out = self.linout(out)

        out = global_add_pool(out, batch)

        return out


