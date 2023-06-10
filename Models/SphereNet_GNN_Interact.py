import torch
from dig.threedgraph.utils import xyz_to_dat
from torch import nn
from torch.nn import ModuleList
from torch_geometric.nn import GINEConv, BatchNorm, global_add_pool, radius_graph
from torch_scatter import scatter

from FeatureDefine.AtomFeature import AtomFeature
from FeatureDefine.BondFeature import BondFeature
from Models.FPN import FPN
from Models.GIN import GIN_Net
from Models.Layers.spherenet import swish, init, update_v, update_u, emb, update_e

import torch_geometric.nn as pnn


class SphereNet_GNN_Interact(torch.nn.Module):
    def __init__(self,
                 energy_and_force=False,
                 num_filters=128,
                 num_gaussians=50, cutoff=5.0, num_layers=4,
                 hidden_channels=128, out_channels=128, int_emb_size=64,
                 basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                 num_spherical=7, num_radial=6, envelope_exponent=5,
                 num_before_skip=1, num_after_skip=2, num_output_layers=3,
                 act=swish, output_init='GlorotOrthogonal', use_node_features=True):
        super(SphereNet_GNN_Interact, self).__init__()

        # Sphere_net
        self.cutoff = cutoff
        self.energy_and_force = energy_and_force

        self.init_e = init(num_radial, hidden_channels, act, use_node_features=use_node_features)
        self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init) for _ in
            range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion,
                     num_spherical, num_radial, num_before_skip, num_after_skip, act) for _ in range(num_layers)])

        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])

        self.reset_parameters()

        # GIN
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.nn_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.node_to_hid = nn.Linear(AtomFeature().node_dim, hidden_channels)

        for _ in range(num_layers):
            conv = GINEConv(nn=self.nn_layer, edge_dim=BondFeature().bond_dim)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.lin1 = pnn.Linear(in_channels=hidden_channels,
                               out_channels=config.mid_size)
        self.lin2 = pnn.Linear(in_channels=config.mid_size,
                               out_channels=out_channels)
        self.linout = pnn.Linear(in_channels=out_channels,
                                 out_channels=config.task_num)

        # GNN and FP Net
        self.readout_attention_score = nn.Sequential(
            nn.Linear(config.hidden_size, config.mid_size),
            nn.Tanh(),
            nn.Linear(config.mid_size, 1),
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
        self.gnn_model = GIN_Net(node_channels=AtomFeature().node_dim,
                                 edge_channels=BondFeature().bond_dim,
                                 hidden_channels=config.hidden_size,
                                 out_channels=config.out_size_graph,
                                 num_layers=config.num_layers,
                                 num_classes=config.task_num)
        self.fcn_model = FPN(in_size=config.fingerprints_size,
                             hidden_size=config.hidden_size,
                             out_size=config.out_size_graph,
                             drop_p=config.drop_p,
                             device=config.device).to(config.device)
        self.gcn_lin1 = nn.Linear(config.out_size_graph, config.out_size_graph)
        self.fcn_lin1 = nn.Linear(config.out_size_graph, config.out_size_graph)
        if config.use_rdkit_feature:
            self.hid_lin = nn.Linear(
                config.out_size_graph + config.out_size_graph + config.out_size_sch + config.rdkit_feature_size,
                config.mid_size)
        else:
            self.hid_lin = nn.Linear(config.out_size_graph + config.out_size_graph + config.out_size_sch,
                                     config.hidden_size)
        self.out_lin = nn.Linear(config.hidden_size, config.task_num)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()

    def forward(self, batch_data):
        X = batch_data.to(config.device)
        # schnet
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()
        # print(f'pos[0].shape = {pos[0].shape} pos[1].shape = {pos[1].shape}')
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes = z.size(0)
        sphere_dist, sphere_angle, sphere_torsion, sphere_i, sphere_j, sphere_idx_kj, sphere_idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True)
        if torch.isnan(sphere_dist).any() or torch.isnan(sphere_angle).any() or torch.isnan(sphere_torsion).any():
            raise Exception('发现 nan')

        emb = self.emb(sphere_dist, sphere_angle, sphere_torsion, sphere_idx_kj)

        # Initialize edge, node, graph features
        e = self.init_e(z, emb, sphere_i, sphere_j)
        v = self.init_v(e, sphere_i)
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch)  # scatter(v, batch, dim=0)

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, sphere_idx_kj, sphere_idx_ji)
            v = update_v(e, sphere_i)
            u = update_u(u, v, batch)  # u += scatter(v, batch, dim=0)

        # print(u)
        sphere_out = v
        sphere_e = e
        sphere_u = u

        # GNN Model
        hid = self.node_to_hid(X.x)
        # print(f'gnn_input.shape = {hid.shape}')
        for i in range(len(self.convs) - 1):
            conv, batch_norm = self.convs[i], self.batch_norms[i]
            hid = torch.relu(conv(x=hid, edge_index=X.edge_index, edge_attr=X.edge_attr))
        gnn_out = hid

        # FCN Model
        fcn_out = self.fcn_model(X)

        # 交互
        # sch_out, gnn_out = torch.concat([sch_out, gnn_out], dim=1), torch.concat([gnn_out, sch_out], dim=1)
        sphere_out = sphere_out + gnn_out
        gnn_out = sphere_out
        # 最后一轮传播 sphere_net
        sphere_e = self.update_es[-1](sphere_e, emb, sphere_idx_kj, sphere_idx_ji)
        sphere_v = self.update_vs[-1](sphere_e, sphere_i)
        sphere_out = self.update_us[-1](sphere_u, sphere_v, batch)
        # 最后一轮传播 gnn
        conv, batch_norm = self.convs[-1], self.batch_norms[-1]
        gnn_out = torch.relu(conv(x=gnn_out, edge_index=X.edge_index, edge_attr=X.edge_attr))
        gnn_out = global_add_pool(gnn_out, X.batch)

        if config.use_rdkit_feature:
            gnn_out = torch.concat([gnn_out, X.rdkit_feature.view(-1, config.rdkit_feature_size)], dim=1)

        out = torch.concat([gnn_out, fcn_out, sphere_out], dim=1)
        out = self.hid_lin(out)
        out = self.relu(out)
        out = self.out_lin(out)

        # if not self.training:
        #     out = self.sigmoid(out)

        if len(out.shape) != 2:
            out = out.unsqueeze(1)
        return out, gnn_out, sphere_out
