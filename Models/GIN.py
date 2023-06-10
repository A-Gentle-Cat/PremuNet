from torch_geometric.nn import BatchNorm, GINEConv, global_add_pool, global_mean_pool, global_max_pool
from torch.nn import ModuleList
from torch_geometric.utils import unbatch


from torch_geometric import nn as pnn

import config
from FeatureDefine.MolFeature import MolFeature
from Models.Loss.MV_GNN_MSE import *
from Models.CrossAttention import CrossAttention
from Models.Layers.Readout import GlobalAttention
from Models.Layers.attention import Attention


class GIN_Net(nn.Module):
    def __init__(self,
                 node_channels,
                 edge_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 predict=False,
                 num_classes=None):
        super().__init__()

        self.predict = predict
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.compress_pre_x = nn.Linear(config.pretrained_atom_feature_size, 64)
        self.nn_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        if config.use_pretrained_atom_feature:
            self.node_to_hid = nn.Linear(node_channels+256, hidden_channels)
        else:
            self.node_to_hid = nn.Linear(node_channels, hidden_channels)
        self.attn_aggr = Attention(in_feature=out_channels,
                                   hidden=config.gnn_hidden_channels,
                                   out_feature=out_channels)

        for _ in range(num_layers):
            conv = GINEConv(nn=self.nn_layer, edge_dim=MolFeature().get_bond_dim())
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.lin1 = pnn.Linear(in_channels=hidden_channels,
                               out_channels=config.gnn_hidden_channels)
        self.lin2 = pnn.Linear(in_channels=config.gnn_hidden_channels,
                               out_channels=out_channels)
        if self.predict:
            self.linout = pnn.Linear(in_channels=out_channels,
                                     out_channels=num_classes)

        if config.aggr_type == 'attention':
            self.readout = GlobalAttention(in_feature=out_channels,
                                           hidden_size=config.gnn_hidden_channels,
                                           out_feature=config.attn_outsize)
            self.att_lin1 = nn.Linear(in_features=config.attn_outsize * out_channels,
                                      out_features=config.gnn_hidden_channels)
            self.att_lin2 = nn.Linear(in_features=config.gnn_hidden_channels,
                                      out_features=out_channels)
        elif config.aggr_type == 'sum':
            self.readout = global_add_pool
        elif config.aggr_type == 'mean':
            self.readout = global_mean_pool
        elif config.aggr_type == 'max':
            self.readout = global_max_pool
        else:
            self.readout = global_add_pool

        self.device = config.device

    def forward(self, X):
        edge_index = X.edge_index
        if config.use_pretrained_atom_feature:
            # print(f'x.shape: {X.x.shape} prex.shape: {X.pre_x.shape}')
            # compressed_pre_x = self.compress_pre_x(X.pre_x)
            node_feature = torch.concat([X.x, X.pre_x], dim=1)
        else:
            node_feature = X.x
        edge_feature = X.edge_attr
        edge_index = edge_index.to(self.device)
        node_feature = node_feature.to(self.device)
        batch = X.batch.to(self.device)
        edge_feature = edge_feature.to(self.device)

        # def forward(self, x: Tensor, edge_index: Adj,
        #             edge_attr: OptTensor = None) -> Tensor:
        hid = self.node_to_hid(node_feature)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            hid = torch.relu(conv(x=hid, edge_index=edge_index, edge_attr=edge_feature))

        out = self.lin1(hid)
        out = torch.relu(out)
        out = self.lin2(out)
        # out = torch.relu(out)
        # out = self.linout(out)

        out = self.readout(out, batch)
        if config.aggr_type == 'attention':
            out = self.att_lin2(torch.relu(self.att_lin1(out)))
        # out,  = self.attn_aggr(out)
        if self.predict:
            out = torch.relu(out)
            out = self.linout(out)

        return out


class GIN_Cross_Attention_Net(nn.Module):
    def __init__(self,
                 node_channels,
                 edge_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 nheads,
                 predict=False,
                 num_classes=None):
        super().__init__()

        self.predict = predict
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.nn_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.node_to_hid = nn.Linear(node_channels, hidden_channels)
        self.attn_aggr = Attention(in_feature=out_channels,
                                   hidden=config.hidden_size,
                                   out_feature=out_channels)

        for _ in range(num_layers):
            conv = GINEConv(nn=self.nn_layer, edge_dim=MolFeature().get_bond_dim())
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.lin1 = pnn.Linear(in_channels=hidden_channels,
                               out_channels=config.mid_size)
        self.lin2 = pnn.Linear(in_channels=config.mid_size,
                               out_channels=out_channels)

        if config.aggr_type == 'attention':
            self.readout = GlobalAttention(in_feature=out_channels,
                                           hidden_size=config.hidden_size,
                                           out_feature=config.attn_outsize)
            self.att_lin1 = nn.Linear(in_features=config.attn_outsize * out_channels,
                                      out_features=config.mid_size)
            self.att_lin2 = nn.Linear(in_features=config.mid_size,
                                      out_features=out_channels)
        elif config.aggr_type == 'sum':
            self.readout = global_add_pool
        elif config.aggr_type == 'mean':
            self.readout = global_mean_pool
        elif config.aggr_type == 'max':
            self.readout = global_max_pool
        else:
            self.readout = global_add_pool

        self.cross_attention = CrossAttention(input_size=out_channels,
                                              attn_hidden=128)

        if self.predict:
            self.linout = pnn.Linear(in_channels=128,
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

        hid = self.node_to_hid(node_feature)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            hid = torch.relu(conv(x=hid, edge_index=edge_index, edge_attr=edge_feature))

        out = self.lin1(hid)
        out = torch.relu(out)
        out = self.lin2(out)

        gnn_atom_feature = unbatch(out, batch)
        tsfm_atom_feature = unbatch(out, batch)
        out = []
        for gnn_fea, tsfm_fea in zip(gnn_atom_feature, tsfm_atom_feature):
            # print(gnn_fea.shape, tsfm_fea.shape)
            out.append(self.cross_attention(tsfm_fea, gnn_fea))
        out = torch.concat(out, dim=0)
        # print(f'out.shape = {out.shape}')
        # out = self.cross_attention(gnn_atom_feature, tsfm_atom_feature)

        out = self.readout(out, batch)
        if config.aggr_type == 'attention':
            out = self.att_lin2(torch.relu(self.att_lin1(out)))

        if self.predict:
            out = self.linout(out)

        return out
