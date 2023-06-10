from torch.nn import Embedding
from torch_geometric.nn import radius_graph

from FeatureDefine.AtomFeature import AtomFeature
from FeatureDefine.BondFeature import BondFeature
from Models.FPN import FPN
from Models.GIN import GIN_Net
from Models.Layers.schnet import emb, update_v, update_e, update_u
from torch_geometric.nn import BatchNorm, GINEConv
from torch.nn import ModuleList

from torch_geometric import nn as pnn
from Models.Loss.MV_GNN_MSE import *
from LoadData.HIV import *


class SchNet_GNN_Interact(torch.nn.Module):
    def __init__(self,
                 energy_and_force=False,
                 cutoff=10.0,
                 num_layers=6,
                 hidden_channels=128,
                 out_channels=128,
                 num_filters=128,
                 num_gaussians=50):
        super(SchNet_GNN_Interact, self).__init__()

        # Schnet
        self.energy_and_force = energy_and_force
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians

        self.init_v = Embedding(100, hidden_channels)
        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        self.update_vs = torch.nn.ModuleList([update_v(hidden_channels, num_filters) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, num_filters, num_gaussians, cutoff) for _ in range(num_layers)])

        self.update_u = update_u(hidden_channels, out_channels)

        self.reset_parameters()

        #GIN
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
                               out_channels=mid_size)
        self.lin2 = pnn.Linear(in_channels=mid_size,
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
            self.hid_lin = nn.Linear(
                config.out_size_graph + config.out_size_graph + config.out_size_sch, config.hidden_size)
        self.out_lin = nn.Linear(config.hidden_size, config.task_num)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def reset_parameters(self):
        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, batch_data):
        X = batch_data.to(config.device)
        # schnet
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist_emb = self.dist_emb(dist)

        v = self.init_v(z)
        for i in range(len(self.update_es)-1):
            update_e, update_v = self.update_es[i], self.update_vs[i]
            e = update_e(v, dist, dist_emb, edge_index)
            v = update_v(v, e, edge_index)
        u = self.update_u(v, batch)
        sch_out = v

        # GNN Model
        hid = self.node_to_hid(X.x)
        # print(f'gnn_input.shape = {hid.shape}')
        for i in range(len(self.convs)-1):
            conv, batch_norm = self.convs[i], self.batch_norms[i]
            hid = torch.relu(conv(x=hid, edge_index=X.edge_index, edge_attr=X.edge_attr))
        gnn_out = hid

        # FCN Model
        fcn_out = self.fcn_model(X)

        # 交互
        # sch_out, gnn_out = torch.concat([sch_out, gnn_out], dim=1), torch.concat([gnn_out, sch_out], dim=1)
        sch_out = sch_out + gnn_out
        gnn_out = sch_out
        # 最后一轮传播 schnet
        update_e, update_v = self.update_es[-1], self.update_vs[-1]
        e = update_e(sch_out, dist, dist_emb, edge_index)
        v = update_v(sch_out, e, edge_index)
        sch_out = self.update_u(v, batch)
        # 最后一轮传播 gnn
        conv, batch_norm = self.convs[-1], self.batch_norms[-1]
        gnn_out = torch.relu(conv(x=gnn_out, edge_index=X.edge_index, edge_attr=X.edge_attr))
        gnn_out = global_add_pool(gnn_out, X.batch)

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
        return out, gnn_out, sch_out

class SchNet_GNN_Interact_before(torch.nn.Module):
    def __init__(self,
                 energy_and_force=False,
                 cutoff=10.0,
                 num_layers=6,
                 hidden_channels=128,
                 out_channels=128,
                 num_filters=128,
                 num_gaussians=50):
        super(SchNet_GNN_Interact_before, self).__init__()

        # Schnet
        self.energy_and_force = energy_and_force
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians

        self.init_v = Embedding(100, hidden_channels)
        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        self.update_vs = torch.nn.ModuleList([update_v(hidden_channels, num_filters) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, num_filters, num_gaussians, cutoff) for _ in range(num_layers)])

        self.update_u = update_u(hidden_channels, out_channels)

        self.reset_parameters()

        #GIN
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
                               out_channels=mid_size)
        self.lin2 = pnn.Linear(in_channels=mid_size,
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
            self.hid_lin = nn.Linear(
                config.out_size_graph + config.out_size_graph + config.out_size_sch, config.hidden_size)
        self.out_lin = nn.Linear(config.hidden_size, config.task_num)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.aggr_2d_3d = nn.Linear(config.out_size_graph + config.out_size_sch, config.out_size_sch)


    def reset_parameters(self):
        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, batch_data):
        X = batch_data.to(config.device)
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()

        # 3d 信息计算
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist_emb = self.dist_emb(dist)

        # 输入初始化
        v = self.init_v(z)
        hid = self.node_to_hid(X.x)
        # print(f'v.shape = {v.shape} hid.shape = {hid.shape}')
        hid = torch.concat([hid, v], dim=1)
        # print(f'hid.shape = {hid.shape}')
        hid = self.aggr_2d_3d(hid)
        v = hid.clone()

        # schnet 传播
        for i in range(len(self.update_es)):
            update_e, update_v = self.update_es[i], self.update_vs[i]
            e = update_e(v, dist, dist_emb, edge_index)
            v = update_v(v, e, edge_index)
        u = self.update_u(v, batch)
        sch_out = v

        # GNN 传播
        # print(f'gnn_input.shape = {hid.shape}')
        for i in range(len(self.convs)):
            conv, batch_norm = self.convs[i], self.batch_norms[i]
            hid = torch.relu(conv(x=hid, edge_index=X.edge_index, edge_attr=X.edge_attr))
        gnn_out = hid

        # FCN Model
        fcn_out = self.fcn_model(X)

        if config.use_rdkit_feature:
            gnn_out = torch.concat([gnn_out, X.rdkit_feature.view(-1, config.rdkit_feature_size)], dim=1)

        gnn_out = global_add_pool(gnn_out, X.batch)
        sch_out = self.update_u(v, X.batch)

        # print(f'schout.shape = {sch_out.shape} gnnout.shape = {gnn_out.shape} fcnout.shape = {fcn_out.shape}')
        out = torch.concat([gnn_out, fcn_out, sch_out], dim=1)
        out = self.hid_lin(out)
        out = self.relu(out)
        out = self.out_lin(out)

        # if not self.training:
        #     out = self.sigmoid(out)

        if len(out.shape) != 2:
            out = out.unsqueeze(1)
        return out, gnn_out, sch_out

