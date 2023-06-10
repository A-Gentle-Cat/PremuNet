import os

import config
from Models.Layers.GraphSequenceAttention import GraphSequenceAttention
from Utils.get_models import get_gnn_model
from Models.FPN import FPN
from Models.MV_GNN import *


class FP_GNN_NET(nn.Module):
    def __init__(self, predictor=False):
        super(FP_GNN_NET, self).__init__()
        # self.gnn_model = get_gnn_model('GIN')
        self.gnn_model = get_gnn_model('PNA')
        self.gcn_lin1 = nn.Linear(config.gnn_out_channels, config.gnn_out_channels)

        if config.fingerprints:
            self.fpn_model = FPN(in_channals=config.fingerprints_size,
                                 hidden_channals=config.fpn_hidden_channels,
                                 mid_channals=config.fpn_mid_channels,
                                 out_channals=config.fpn_out_channels,
                                 drop_p=config.gnn_dropout)
            self.fcn_lin1 = nn.Linear(config.fpn_out_channels, config.fpn_out_channels)
        if config.use_rdkit_feature:
            self.hid_lin = nn.Linear(config.gnn_out_channels + config.fpn_out_channels + config.rdkit_feature_size, config.fcn_mid_channels)
        else:
            self.hid_lin = nn.Linear(config.gnn_out_channels + config.fpn_out_channels, config.fcn_mid_channels)
        if not predictor:
            self.out_lin = nn.Linear(config.fcn_mid_channels, config.fcn_out_channels)
        else:
            self.out_lin = nn.Linear(config.fcn_mid_channels, config.task_num)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.predictor = predictor
        self.drop = nn.Dropout(config.gnn_dropout)

    def forward(self, X):
        X = X.to(config.device)
        gnn_out = self.gnn_model(X)

        if config.fingerprints:
            fpn_out = self.fpn_model(X)
            fpn_out = self.fcn_lin1(fpn_out)
            gnn_out = torch.concat([gnn_out, fpn_out], dim=1)
        if config.use_rdkit_feature:
            rdkit_2d_out = X.rdkit_feature.reshape(-1, config.rdkit_feature_size)
            gnn_out = torch.concat([gnn_out, rdkit_2d_out], dim=1)
        out = gnn_out
        out = self.hid_lin(self.relu(out))
        out = self.relu(out)
        out = self.drop(out)

        out = self.out_lin(out)

        return out

class FP_GNN_CrossAttention_NET(nn.Module):
    def __init__(self,
                 node_channels,
                 edge_channels,
                 gnn_hidden_channels,
                 gnn_out_channels,
                 nheads,
                 fcn_in_channels,
                 fcn_out_channels,
                 fcn_hidden_channels,
                 mid_hidden_size,
                 num_classes,
                 drop_p,
                 device):
        super(FP_GNN_CrossAttention_NET, self).__init__()
        self.readout_attention_score = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.Tanh(),
            nn.Linear(mid_size, 1),
            nn.Softmax(dim=1)
        )
        self.gnn_model = get_gnn_model('GIN')
        self.gcn_lin1 = nn.Linear(gnn_out_channels, gnn_out_channels)

        if config.fingerprints:
            self.fcn_model = FPN(fcn_in_channels,
                                 fcn_hidden_channels,
                                 mid_size,
                                 fcn_out_channels,
                                 drop_p=drop_p,
                                 device=device).to(device)
            self.fcn_lin1 = nn.Linear(fcn_out_channels, fcn_out_channels)
        if config.use_rdkit_feature:
            self.hid_lin = nn.Linear(gnn_out_channels + fcn_out_channels + config.rdkit_feature_size, mid_hidden_size)
        else:
            self.hid_lin = nn.Linear(gnn_out_channels + fcn_out_channels, mid_hidden_size)
        self.out_lin = nn.Linear(mid_hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(config.drop_p)

    def forward(self, X):
        X = X.to(config.device)
        gnn_out = self.gnn_model(X)

        if config.fingerprints:
            fcn_out = self.fcn_model(X)
            fcn_out = self.fcn_lin1(fcn_out)
            gnn_out = torch.concat([gnn_out, fcn_out], dim=1)
        if config.use_rdkit_feature:
            rdkit_2d_out = X.rdkit_feature.reshape(-1, config.rdkit_feature_size)
            gnn_out = torch.concat([gnn_out, rdkit_2d_out], dim=1)
        out = gnn_out
        out = self.hid_lin(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.out_lin(out)

        # if not self.training:
        #     out = self.sigmoid(out)

        return out

