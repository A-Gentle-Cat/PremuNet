from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch
import config

from torch_geometric import nn as pnn

from Models.GCN_Net import *
from Models.Layers.GraphAttentionReadout import GraphAttentionReadout


class MV_GNN_Graph(nn.Module):
    def __init__(self,
                 node_size,
                 edge_size,
                 hidden_size,
                 mid_size,
                 out_size,
                 num_layers,
                 drop_p,
                 device=torch.device('cpu'),
                 aggr='add'):
        super(MV_GNN_Graph, self).__init__()
        self.node_to_hidden = pnn.Linear(node_size, hidden_size)
        self.message_to_hidden = pnn.Linear(hidden_size * 2 + edge_size,
                                            hidden_size)
        self.mpn_encoder = MPN(message_to_hidden=self.message_to_hidden,
                               aggr=aggr)

        self.num_layers = num_layers
        self.h0 = None
        self.outMsg = MV_GNN_outMsg()
        self.device = device
        self.hid_feat = None

        self.midLin = nn.Linear(in_features=hidden_size + node_size, out_features=mid_size)
        self.outMsgLin = nn.Linear(in_features=hidden_size*2 + node_size, out_features=hidden_size)
        self.dropout = nn.Dropout(p=drop_p)

        self.msg_attention_score = nn.Sequential(
            nn.Linear(hidden_size * 2 + edge_size, mid_size),
            nn.Tanh(),
            nn.Linear(mid_size, 1),
            nn.Softmax(dim=1)
        )
        self.msg_aggr = nn.MultiheadAttention(hidden_size * 2 + edge_size, num_heads=1)

        self.outLin = nn.Linear(in_features=hidden_size, out_features=out_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.node_to_hidden.reset_parameters()
        self.message_to_hidden.reset_parameters()
        self.midLin.reset_parameters()
        self.outLin.reset_parameters()

    def forward(self, X):
        edge_index = X.edge_index.to(self.device)
        x = X.x.to(self.device)
        edge_attr = X.edge_attr.to(self.device)
        batch = X.batch.to(self.device)

        self.h0 = self.node_to_hidden(x)
        self.h0 = torch.relu(self.h0)

        self.hid_feat = self.h0
        edge_index_with_self, edge_attr_with_self = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=len(x))
        # edge_index_with_self, edge_attr_with_self = edge_index, edge_attr
        self.hid_feat = self.mpn_encoder(self.hid_feat, edge_attr_with_self, edge_index_with_self, self.h0)

        out = self.outMsg(edge_index_with_self, x, self.hid_feat)
        out = self.outMsgLin(out)
        out = torch.relu(out)

        # out = self.attention_readout(out, index=batch)
        out = global_add_pool(out, batch)
        out = self.outLin(out)

        return out


class MV_GNN_LineGraph(MessagePassing):
    def __init__(self, node_size, edge_size, hidden_size, mid_size, out_size, num_layers, device=torch.device('cpu'),
                 aggr='add'):
        super(MV_GNN_LineGraph, self).__init__(aggr=aggr)
        self.node_to_hidden = pnn.Linear(edge_size, hidden_size)
        self.message_to_hidden = pnn.Linear(hidden_size * 2 + node_size,
                                            hidden_size)

        self.num_layers = num_layers
        self.h0 = None
        self.outMsg = MV_GNN_outMsg()
        self.device = device
        self.hid_feat = None

        self.midLin = nn.Linear(in_features=hidden_size + edge_size, out_features=mid_size)

        self.attention_score = nn.Sequential(
            nn.Linear(hidden_size + edge_size, mid_size),
            nn.Tanh(),
            nn.Linear(mid_size, 1),
            nn.Softmax(dim=1)
        )

        self.attention_readout = AttentionalAggregation(gate_nn=self.attention_score)
        self.graph_readout = GraphAttentionReadout(in_dim=out_size, att_dim=config.att_dim, r_dim=1)
        self.outLin = nn.Linear(in_features=hidden_size + edge_size, out_features=out_size)

        self.flatten = nn.Flatten()

        self.reset_parameters()

    def reset_parameters(self):
        self.node_to_hidden.reset_parameters()
        self.message_to_hidden.reset_parameters()
        self.midLin.reset_parameters()
        self.outLin.reset_parameters()

    def forward(self, X):
        edge_index = X.edge_index.to(self.device)
        x = X.x.to(self.device)
        edge_attr = X.edge_attr.to(self.device)
        batch = X.batch.to(self.device)

        self.h0 = self.node_to_hidden(x)
        self.h0 = torch.relu(self.h0)

        self.hid_feat = self.h0
        for _ in range(self.num_layers):
            self.hid_feat = self.propagate(edge_index=edge_index,
                                           x=self.hid_feat,
                                           edge_attr=edge_attr)
        out = self.outMsg(edge_index, x, self.hid_feat)
        # out = self.attention_readout(out, index=batch)
        out = global_mean_pool(out, batch)
        out = self.outLin(out)
        # out = self.graph_readout(out, batch)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr) -> Tensor:
        # print(x_i.shape, x_j.shape, edge_attr.shape)
        out = torch.concat([x_i, x_j, edge_attr], dim=1)
        # print('out.shape', out.shape)
        return out

    def update(self, inputs: Tensor) -> Tensor:
        out = self.message_to_hidden(inputs) + self.h0
        out = torch.relu(out)
        return out


class MPN(MessagePassing):
    def __init__(self, message_to_hidden: nn.Module, aggr):
        super(MPN, self).__init__(aggr=aggr)
        self.message_to_hidden = message_to_hidden
        self.h0 = None

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr) -> Tensor:
        # print(x_i.shape, x_j.shape, edge_attr.shape)
        out = torch.concat([x_i, x_j, edge_attr], dim=1)
        # out, _ = self.msg_aggr(out, out, out, need_weights=False)
        # print('out.shape', out.shape)
        return out

    def update(self, inputs: Tensor) -> Tensor:
        out = self.message_to_hidden(inputs) + self.h0
        out = torch.relu(out)
        return out

    def forward(self, hid_feat, edge_attr, edge_index, h0):
        self.h0 = h0
        for _ in range(config.num_layers):
            hid_feat = self.propagate(edge_index=edge_index,
                                      x=hid_feat,
                                      edge_attr=edge_attr)
        return hid_feat


class MV_GNN_outMsg(MessagePassing):
    def __init__(self):
        super(MV_GNN_outMsg, self).__init__(aggr='add')

    def forward(self, edge_index, x, hid):
        return self.propagate(edge_index=edge_index, x=x, h=hid)

    def message(self, x_j: Tensor, h_i: Tensor, h_j: Tensor) -> Tensor:
        out = torch.concat([h_i, h_j, x_j], dim=1)
        return out

    def update(self, inputs: Tensor) -> Tensor:
        return inputs


class MV_GNN_net(nn.Module):
    def __init__(self,
                 node_size,
                 edge_size,
                 hidden_size,
                 mid_size,
                 out_size_graph,
                 out_size_line,
                 ffn_hidden_size,
                 ffn_num_layers,
                 ffn_out_size,
                 num_layers,
                 drop_p,
                 num_classes,
                 device=torch.device('cpu')):
        super(MV_GNN_net, self).__init__()

        self.readout_attention_score = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.Tanh(),
            nn.Linear(mid_size, 1),
            nn.Softmax(dim=1)
        )

        self.graph_mpnn = MV_GNN_Graph(node_size=node_size,
                                       edge_size=edge_size,
                                       hidden_size=hidden_size,
                                       mid_size=mid_size,
                                       out_size=out_size_graph,
                                       num_layers=num_layers,
                                       drop_p=drop_p,
                                       attention_score=self.readout_attention_score,
                                       device=device)
        self.line_mpnn = MV_GNN_Graph(node_size=edge_size,
                                      edge_size=node_size,
                                      hidden_size=hidden_size,
                                      mid_size=mid_size,
                                      out_size=out_size_line,
                                      num_layers=num_layers,
                                      drop_p=drop_p,
                                      attention_score=self.readout_attention_score,
                                      device=device)
        # self.graph_mpnn = MV_GNN_LineGraph(node_size=edge_size,
        #                                    edge_size=node_size,
        #                                    hidden_size=hidden_size,
        #                                    mid_size=mid_size,
        #                                    out_size=out_size_graph,
        #                                    num_layers=num_layers,
        #                                    device=config.device)
        # self.line_mpnn = MV_GNN_LineGraph(node_size=node_size,
        #                                   edge_size=edge_size,
        #                                    hidden_size=hidden_size,
        #                                    mid_size=mid_size,
        #                                    out_size=out_size_line,
        #                                    num_layers=num_layers,
        #                                    device=config.device)

        self.drop_p = drop_p
        self.ffn_hidden_size = ffn_hidden_size
        self.ffn_num_layers = ffn_num_layers
        self.ffn_out_size = ffn_out_size

        self.lin_to_hidden1 = self.create_ffn(out_size_graph, ffn_out_size)
        self.lin_to_hidden2 = self.create_ffn(out_size_line, ffn_out_size)

        if config.use_rdkit_feature:
            self.lin_to_hidden1 = self.create_ffn(out_size_graph + config.rdkit_feature_size, ffn_out_size)
            self.lin_to_hidden2 = self.create_ffn(out_size_line+config.rdkit_feature_size, ffn_out_size)
            self.classifier1 = nn.Linear(ffn_out_size, num_classes)
            self.classifier2 = nn.Linear(ffn_out_size, num_classes)
        else:
            self.lin_to_hidden1 = self.create_ffn(out_size_graph, ffn_out_size)
            self.lin_to_hidden2 = self.create_ffn(out_size_line, ffn_out_size)
            self.classifier1 = nn.Linear(ffn_out_size, num_classes)
            self.classifier2 = nn.Linear(ffn_out_size, num_classes)

        # self.classifier1 = nn.Linear(out_size_graph, num_classes)
        # self.classifier2 = nn.Linear(out_size_line, num_classes)
        self.aggr_class = nn.Linear(num_classes*2, num_classes)

    def create_ffn(self, first_linear_dim, output_size):
        dropout = nn.Dropout(self.drop_p)
        activation = nn.ReLU()
        ffn = [
            dropout,
            nn.Linear(first_linear_dim, self.ffn_hidden_size)
        ]
        for _ in range(self.ffn_num_layers):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size),
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(self.ffn_hidden_size, output_size),
        ])
        return nn.Sequential(*ffn)

    def forward(self, X, X2):
        out1 = self.graph_mpnn(X)
        out2 = self.line_mpnn(X2)

        if config.use_rdkit_feature:
            rdkit_feature = X.rdkit_feature.to(config.device).to(torch.float32)
            rdkit_feature = rdkit_feature.view(-1, config.rdkit_feature_size)
            rdkit_feature = torch.zeros_like(rdkit_feature)
            out1 = torch.concat([out1, rdkit_feature], dim=1)
            out2 = torch.concat([out2, rdkit_feature], dim=1)
        out1 = self.lin_to_hidden1(out1)
        out1 = torch.relu(out1)
        out2 = self.lin_to_hidden2(out2)
        out2 = torch.relu(out2)

        out1 = self.classifier1(out1)
        out2 = self.classifier2(out2)

        output = (out1 + out2) / 2

        # output = self.aggr_class(torch.concat([out1, out2], dim=1))

        return out1, out2, output


class MV_GNN_net_with_fingerprints(nn.Module):
    def __init__(self,
                 node_size,
                 edge_size,
                 hidden_size,
                 mid_size,
                 out_size_graph,
                 out_size_line,
                 ffn_hidden_size,
                 ffn_num_layers,
                 ffn_out_size,
                 num_layers,
                 drop_p,
                 num_classes,
                 device=torch.device('cpu')):
        super(MV_GNN_net_with_fingerprints, self).__init__()

        self.readout_attention_score = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.Tanh(),
            nn.Linear(mid_size, 1),
            nn.Softmax(dim=1)
        )

        self.graph_mpnn = MV_GNN_Graph(node_size=node_size,
                                       edge_size=edge_size,
                                       hidden_size=hidden_size,
                                       mid_size=mid_size,
                                       out_size=out_size_graph,
                                       num_layers=num_layers,
                                       drop_p=drop_p,
                                       attention_score=self.readout_attention_score,
                                       device=device)
        self.line_mpnn = MV_GNN_Graph(node_size=edge_size,
                                      edge_size=node_size,
                                      hidden_size=hidden_size,
                                      mid_size=mid_size,
                                      out_size=out_size_line,
                                      num_layers=num_layers,
                                      drop_p=drop_p,
                                      attention_score=self.readout_attention_score,
                                      device=device)

        self.drop_p = drop_p
        self.ffn_hidden_size = ffn_hidden_size
        self.ffn_num_layers = ffn_num_layers
        self.ffn_out_size = ffn_out_size

        self.lin_to_hidden1 = self.create_ffn(out_size_graph, ffn_out_size)
        self.lin_to_hidden2 = self.create_ffn(out_size_line, ffn_out_size)
        self.classifier1 = nn.Linear(ffn_out_size, num_classes)
        self.classifier2 = nn.Linear(ffn_out_size, num_classes)

    def create_ffn(self, first_linear_dim, output_size):
        dropout = nn.Dropout(self.drop_p)
        activation = nn.ReLU()
        ffn = [
            dropout,
            nn.Linear(first_linear_dim, self.ffn_hidden_size)
        ]
        for _ in range(self.ffn_num_layers):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size),
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(self.ffn_hidden_size, output_size),
        ])
        return nn.Sequential(*ffn)

    def forward(self, X, X2):
        out1 = self.graph_mpnn(X)
        out2 = self.line_mpnn(X2)

        out1 = self.lin_to_hidden1(out1)
        out2 = self.lin_to_hidden2(out2)

        out1 = self.classifier1(out1)
        out2 = self.classifier2(out2)

        output = (out1 + out2) / 2

        return out1, out2, output


# if __name__ == '__main__':
#     device = torch.device('cuda:0')
#     total_dataset = molDataset_BBBP(catogory='train')
#     cur = total_dataset[12]
#     cur.x = torch.tensor([[0, 0, 0],
#                           [0, 0, 1],
#                           [0, 0, 2]], dtype=torch.float32, device=device)
#     cur.edge_attr = torch.tensor([[1., 0., 0., 0., 0., 0.],
#                                   [1., 0., 0., 0., 0., 0.],
#                                   [1., 0., 0., 0., 0., 1.],
#                                   [1., 0., 0., 0., 0., 1.]], device=device)
#     print(cur.edge_index)
#     print(cur.edge_attr)
#     cur_iter = loader.DataLoader([cur], batch_size=1, shuffle=False)
#     for X in cur_iter:
#         out = MV_GNN_Graph(3, 6, 16, 10, 8, 1, device).to(device)(X)
#         print(out.shape)
