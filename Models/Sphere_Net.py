import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import PointNetConv, global_add_pool

from Models.FPN import FPN
from Models.dig.threedgraph.method.schnet import SchNet

from Utils.get_models import get_gnn_model
from config import *


class Sphere_Net(nn.Module):
    def __init__(self,
                 node_channels,
                 edge_channels,
                 gnn_hidden_channels,
                 gnn_out_channels,
                 nheads,
                 fcn_in_channels,
                 fcn_out_channels,
                 fcn_hidden_channels,
                 sch_out_channels,
                 mid_hidden_size,
                 num_classes,
                 drop_p,
                 device):
        super(Sphere_Net, self).__init__()

        self.readout_attention_score = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.Tanh(),
            nn.Linear(mid_size, 1),
            nn.Softmax(dim=1)
        )
        # self.gnn_model = MV_GNN_Graph(node_size=node_channels,
        #                               edge_size=edge_channels,
        #                               hidden_size=hidden_size,
        #                               mid_size=mid_size,
        #                               out_size=out_size_graph,
        #                               num_layers=num_layers,
        #                               drop_p=drop_p,
        #                               attention_score=self.readout_attention_score,
        #                               device=device)
        self.gnn_model = get_gnn_model('GIN')
        self.fcn_model = FPN(fcn_in_channels,
                             fcn_hidden_channels,
                             fcn_out_channels,
                             drop_p=drop_p,
                             device=device).to(device)
        self.gcn_lin1 = nn.Linear(gnn_out_channels, gnn_out_channels)
        self.fcn_lin1 = nn.Linear(fcn_out_channels, fcn_out_channels)
        if config.use_rdkit_feature:
            self.hid_lin = nn.Linear(gnn_out_channels + fcn_out_channels + sch_out_channels + config.rdkit_feature_size, mid_hidden_size)
        else:
            self.hid_lin = nn.Linear(gnn_out_channels + fcn_out_channels + sch_out_channels, mid_hidden_size)
        self.out_lin = nn.Linear(mid_hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.sp_net = SchNet(out_channels=128)

        # self.sp_net = SphereNet(energy_and_force=False, cutoff=10.0, num_layers=4,
        #                   hidden_channels=128, out_channels=64, int_emb_size=64,
        #                   basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        #                   num_spherical=3, num_radial=6, envelope_exponent=5,
        #                   num_before_skip=1, num_after_skip=2, num_output_layers=3)
        # self.sp_net = SchNet(out_channels=config.task_num)
        # self.sp_net = DimeNetPP(out_channels=128)
        # self.sp_net = ComENet(out_channels=128)
        self.sp_lin1 = nn.Linear(64, 32)
        self.sp_lin2 = nn.Linear(32, config.task_num)
        self.act = nn.ReLU()

    def forward(self, X):
        X = X.to(config.device)

        gnn_out = self.gnn_model(X)
        fcn_out = self.fcn_model(X)
        gnn_out = self.gcn_lin1(gnn_out)
        fcn_out = self.fcn_lin1(fcn_out)
        sch_out = self.sp_net(X)
        # sch_out = self.sp_lin2(self.act(self.sp_lin1(sch_out)))
        if config.use_rdkit_feature:
            gnn_out = torch.concat([gnn_out, X.rdkit_feature.view(-1, config.rdkit_feature_size)], dim=1)

        out = torch.concat([gnn_out, fcn_out, sch_out], dim=1)
        out = self.hid_lin(out)
        out = self.relu(out)
        out = self.out_lin(out)

        # if not self.training:
        #     out = self.sigmoid(out)

        if len(out.shape) != 2:
            out = out.unsqueeze(1)
        return out

class Sphere_Net_Contrast(nn.Module):
    def __init__(self,
                 node_channels,
                 edge_channels,
                 gnn_hidden_channels,
                 gnn_out_channels,
                 nheads,
                 fcn_in_channels,
                 fcn_out_channels,
                 fcn_hidden_channels,
                 sch_out_channels,
                 mid_hidden_size,
                 num_classes,
                 drop_p,
                 device):
        super(Sphere_Net_Contrast, self).__init__()

        self.readout_attention_score = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.Tanh(),
            nn.Linear(mid_size, 1),
            nn.Softmax(dim=1)
        )
        # self.gnn_model = MV_GNN_Graph(node_size=node_channels,
        #                               edge_size=edge_channels,
        #                               hidden_size=hidden_size,
        #                               mid_size=mid_size,
        #                               out_size=out_size_graph,
        #                               num_layers=num_layers,
        #                               drop_p=drop_p,
        #                               attention_score=self.readout_attention_score,
        #                               device=device)
        self.gnn_model = get_gnn_model('GIN')
        # self.gnn_model = GIN_Net(node_channels=node_channels,
        #                          edge_channels=edge_channels,
        #                          hidden_channels=config.hidden_size,
        #                          out_channels=config.out_size_graph,
        #                          num_layers=config.num_layers,
        #                          num_classes=config.task_num)
        self.fcn_model = FPN(fcn_in_channels,
                             fcn_hidden_channels,
                             fcn_out_channels,
                             drop_p=drop_p,
                             device=device).to(device)
        self.gcn_lin1 = nn.Linear(gnn_out_channels, gnn_out_channels)
        self.fcn_lin1 = nn.Linear(fcn_out_channels, fcn_out_channels)
        if config.use_rdkit_feature:
            self.hid_lin = nn.Linear(gnn_out_channels + fcn_out_channels + sch_out_channels + config.rdkit_feature_size, mid_hidden_size)
        else:
            self.hid_lin = nn.Linear(gnn_out_channels + fcn_out_channels + sch_out_channels, mid_hidden_size)
        self.node_to_hid = nn.Linear(node_channels, mid_hidden_size)
        self.out_lin = nn.Linear(mid_hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # self.sp_net = SchNet(out_channels=128)
        self.sp_net = ModuleList()
        self.sp_to_hid = nn.Linear(131, mid_hidden_size)
        for _ in range(3):
            self.sp_net.append(PointNetConv(global_nn=self.sp_to_hid))

        # self.sp_net = SphereNet(energy_and_force=False, cutoff=10.0, num_layers=4,
        #                   hidden_channels=128, out_channels=128, int_emb_size=64,
        #                   basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        #                   num_spherical=3, num_radial=6, envelope_exponent=5,
        #                   num_before_skip=1, num_after_skip=2, num_output_layers=3)
        # self.sp_net = SchNet(out_channels=config.task_num)
        # self.sp_net = DimeNetPP(out_channels=256)
        # self.sp_net = ProNet(out_channels=config.task_num)
        # self.sp_net = ComENet(out_channels=256)
        self.sp_lin1 = nn.Linear(128, 64)
        self.sp_lin2 = nn.Linear(64, config.task_num)
        self.act = nn.ReLU()

    def forward(self, X):
        X = X.to(config.device)

        gnn_out = self.gnn_model(X)
        fcn_out = self.fcn_model(X)
        gnn_out = self.gcn_lin1(gnn_out)
        fcn_out = self.fcn_lin1(fcn_out)

        sch_out = self.node_to_hid(X.x)
        for layers in self.sp_net:
            sch_out = layers(x=sch_out, pos=X.pos, edge_index=X.edge_index)
        sch_out = global_add_pool(sch_out, batch=X.batch)

        out_gnn = gnn_out
        out_sch = sch_out
        if config.use_rdkit_feature:
            gnn_out = torch.concat([gnn_out, X.rdkit_feature.view(-1, config.rdkit_feature_size)], dim=1)

        out = torch.concat([gnn_out, fcn_out, sch_out], dim=1)
        out = self.hid_lin(out)
        out = self.relu(out)
        out = self.out_lin(out)

        # if not self.training:
        #     out = self.sigmoid(out)

        if len(out.shape) != 2:
            out = out.unsqueeze(1)
        return out, out_gnn, out_sch

class Sch_Net(nn.Module):
    def __init__(self,
                 gnn_out_channels,
                 fcn_out_channels,
                 sch_out_channels,
                 mid_hidden_size,
                 num_classes):
        super(Sch_Net, self).__init__()
        if config.use_rdkit_feature:
            self.hid_lin = nn.Linear(gnn_out_channels + fcn_out_channels + sch_out_channels + config.rdkit_feature_size, mid_hidden_size)
        else:
            self.hid_lin = nn.Linear(gnn_out_channels + fcn_out_channels + sch_out_channels, mid_hidden_size)
        self.out_lin = nn.Linear(mid_hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.sp_net = SchNet(out_channels=64)

        # self.sp_net = SphereNet(energy_and_force=False, cutoff=10.0, num_layers=4,
        #                   hidden_channels=128, out_channels=64, int_emb_size=64,
        #                   basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        #                   num_spherical=3, num_radial=6, envelope_exponent=5,
        #                   num_before_skip=1, num_after_skip=2, num_output_layers=3)
        # self.sp_net = SchNet(out_channels=config.task_num)
        # self.sp_net = DimeNetPP(out_channels=64)
        # self.sp_net = ComENet(out_channels=64)
        self.sp_lin1 = nn.Linear(64, 32)
        self.sp_lin2 = nn.Linear(32, config.task_num)
        self.act = nn.ReLU()

    def forward(self, X):
        X = X.to(config.device)
        sch_out = self.sp_net(X)
        out = self.sp_lin1(sch_out)
        out = self.relu(out)
        out = self.sp_lin2(out)

        # if not self.training:
        #     out = self.sigmoid(out)

        if len(out.shape) != 2:
            out = out.unsqueeze(1)
        return out

