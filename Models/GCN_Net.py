import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, AttentionalAggregation
import torch_geometric.nn as pnn
import torch.nn.functional as F
import config

from Models.GCN_layer import GCN


class molGCN_only_node(nn.Module):
    def __init__(self, in_channels, hidden_channel, out_channels, num_classes, device):
        super(molGCN_only_node, self).__init__()
        self.gcn1 = GCN(in_size=in_channels, out_size=hidden_channel)
        self.relu = nn.ReLU()
        self.gcn2 = GCN(hidden_channel, out_channels)
        self.lin = nn.Linear(out_channels, num_classes)
        self.droupout = nn.Dropout(0.3)
        self.device = device

    def forward(self, X):
        edge_index = X.edge_index
        node_feature = X.x
        edge_feature = X.edge_attr
        edge_index = edge_index.to(self.device)
        node_feature = node_feature.to(self.device)
        batch = X.batch.to(self.device)
        edge_feature = edge_feature.to(self.device)

        hidden_feature = self.gcn1(node_feature, edge_index)
        hidden_feature = self.relu(hidden_feature)
        hidden_feature = self.gcn2(hidden_feature, edge_index)

        # print('hidden.shape: ', hidden_feature.shape)
        out = global_mean_pool(hidden_feature, batch)
        out = self.droupout(out)
        out = self.lin(out)

        return out


class molGCN_GAT(nn.Module):
    def __init__(self,
                 node_channels,
                 edge_channels,
                 hidden_channel,
                 out_channels,
                 nheads,
                 device,
                 num_classes=None,
                 drop_p=0.0,
                 classifier=True):
        super(molGCN_GAT, self).__init__()
        self.device = device
        self.gat_conv1 = pnn.GATConv(in_channels=node_channels,
                                     out_channels=hidden_channel,
                                     heads=nheads,
                                     dropout=drop_p,
                                     edge_dim=edge_channels)
        self.gat_conv2 = pnn.GATConv(in_channels=hidden_channel*nheads,
                                     out_channels=out_channels,
                                     heads=1,
                                     dropout=drop_p,
                                     edge_dim=edge_channels)

        self.attention_score = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )

        self.attention_readout = AttentionalAggregation(gate_nn=self.attention_score)

        self.drop = nn.Dropout(p=drop_p)
        self.elu = nn.ELU()
        self.classifier = classifier

        if classifier:
            self.lin = pnn.Linear(out_channels, num_classes)

    def forward(self, X):
        edge_index = X.edge_index
        node_feature = X.x
        edge_feature = X.edge_attr
        edge_index = edge_index.to(self.device)
        node_feature = node_feature.to(self.device)
        batch = X.batch.to(self.device)
        edge_feature = edge_feature.to(self.device)
        # edge_feature = None

        hidden_feature = self.gat_conv1(node_feature, edge_index, edge_feature)
        hidden_feature = self.drop(hidden_feature)
        hidden_feature = self.gat_conv2(hidden_feature, edge_index, edge_feature)
        hidden_feature = self.elu(hidden_feature)

        hidden_feature = F.log_softmax(hidden_feature, dim=1)
        # out = self.attention_readout(hidden_feature, batch)
        out = global_mean_pool(hidden_feature, batch)
        out = self.drop(out)

        if self.classifier:
            out = self.lin(out)

        return out


class molGCN_NNConv(nn.Module):
    def __init__(self, node_channels, edge_channels, hidden_channel1, hidden_channel2, out_channels, drop_p, num_classes):
        super(molGCN_NNConv, self).__init__()
        self.edge_lin1 = nn.Linear(edge_channels, node_channels * hidden_channel1)
        self.nnconv1 = pnn.NNConv(node_channels, hidden_channel1, flow='source_to_target', nn=self.edge_lin1)
        self.relu = nn.ReLU()

        self.edge_lin2 = nn.Linear(edge_channels, hidden_channel1 * hidden_channel2)
        self.nnconv2 = pnn.NNConv(hidden_channel1, hidden_channel2, flow='source_to_target', nn=self.edge_lin2)
        self.relu2 = nn.ReLU()

        self.edge_lin3 = nn.Linear(edge_channels, hidden_channel2 * out_channels)
        self.nnconv3 = pnn.NNConv(hidden_channel2, out_channels, flow='source_to_target', nn=self.edge_lin3)
        self.relu3 = nn.ReLU()

        self.droupout = nn.Dropout(drop_p)
        self.num_classes = num_classes
        if num_classes is not None:
            self.lin = nn.Linear(out_channels, num_classes)

    def forward(self, X):
        edge_index = X.edge_index.to(config.device)
        node_feature = X.x.to(config.device)
        edge_feature = X.edge_attr.to(config.device)
        batch = X.batch.to(config.device)
        hidden_feature = self.nnconv1(node_feature, edge_index, edge_feature)
        hidden_feature = self.relu(hidden_feature)
        hidden_feature = self.nnconv2(hidden_feature, edge_index, edge_feature)
        hidden_feature = self.relu2(hidden_feature)
        hidden_feature = self.nnconv3(hidden_feature, edge_index, edge_feature)
        hidden_feature = self.relu3(hidden_feature)

        out = global_add_pool(hidden_feature, batch)
        # out = Set2Set(hidden_feature, batch)
        out = self.droupout(out)

        if self.num_classes is not None:
            out = self.lin(out)

        return out
