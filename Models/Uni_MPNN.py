import torch

import config
from Models.FP_GNN_NET import FP_GNN_NET
from Models.Layers.attention import Attention
from UnifiedMolPretrain.pretrain3d.model.gnn import GNNet
from UnifiedMolPretrain.pretrain3d.utils.misc import PreprocessBatch
from Utils.get_models import get_gnn_model
from config import *
from Models.FPN import FPN
from Models.MV_GNN import *


class FPN_Part(nn.Module):
    def __init__(self, in_channals, hidden_channals, mid_channals, out_channals, drop_p):
        super(FPN_Part, self).__init__()
        self.in_channals = in_channals
        self.lin1 = nn.Linear(in_channals, hidden_channals)
        self.lin2 = nn.Linear(hidden_channals, mid_channals)
        self.lin3 = nn.Linear(mid_channals, out_channals)
        # self.drop = nn.Dropout(p=drop_p)
        self.relu = nn.ReLU()

    def forward(self, X):
        # (batchsize, fingerprints size)
        if config.concat_two_fingerprints:
            fp = torch.concat([X.tsfm_fp.view(-1, config.fingerprints_size_trans),
                               X.traditional_fp.view(-1, config.fingerprints_size_ecfp)], dim=1).to(config.device)
        elif config.fingerprints_catogory == 'trans':
            fp = X.tsfm_fp.to(config.device)
        else:
            fp = X.traditional_fp.to(config.device)
        # print(f'tsfm: {X.tsfm_fp.view(-1, config.fingerprints_size_trans).shape} tradi: {X.traditional_fp.view(-1, config.fingerprints_size_ecfp).shape}')
        # print(f'fp.shape: {fp.shape}')
        fp = fp.view(-1, self.in_channals)
        # print('================================================')
        # print(torch.isnan(fp).any())
        # print(torch.isnan(self.lin1.weight.data).any())
        # print(torch.max(self.lin1.weight.data))
        # print(torch.min(self.lin1.weight.data))
        # print(torch.mean(self.lin1.weight.data))
        hidden_feature = self.lin1(fp)
        hidden_feature = self.relu(hidden_feature)
        out = torch.relu(self.lin2(hidden_feature))
        out = self.lin3(out)

        return out


class Uni_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcgnn_model = FP_GNN_NET(predictor=False)
        self.unified_model = GNNet(graph_pooling=config.graph_pooling,
                                  global_reducer=config.global_reducer,
                                  node_reducer=config.node_reducer,
                                  dropout=config.unified_dropout,
                                   raw_with_pos=config.raw_with_pos,
                                   num_tasks=config.unified_out_channels)
        self.act = torch.nn.ReLU()
        # self.ffn_out = nn.Sequential(
        #     nn.Linear(config.fcn_out_channels + config.unified_out_channels, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, config.task_num)
        # )
        if config.agg_method == 'concat':
            self.ffn_out = nn.Linear(config.fcn_out_channels + config.unified_out_channels, config.task_num)
        else:
            assert config.fcn_out_channels == config.unified_out_channels
            self.ffn_out = nn.Linear(config.fcn_out_channels, config.task_num)

        self.processor = PreprocessBatch(True, False)

        if config.use_weight_from_file:
            checkpoint = torch.load(config.uni_checkpoint_path, map_location=torch.device("cpu"))[
                "model_state_dict"
            ]
            self.unified_model.load_state_dict(checkpoint)

    def forward(self, X):
        # self.processor.process(X)
        out_fcgnn = self.fcgnn_model(X)
        out_unified, _, _, _ = self.unified_model(X)

        # out_fcgnn = self.fcgnn_norm(out_fcgnn)
        # out_unified = self.unified_norm(out_unified)

        if config.agg_method == 'concat':
            agg_out = torch.concat([out_fcgnn, out_unified], dim=1)
        elif config.agg_method == 'add':
            agg_out = out_fcgnn + out_unified
        elif config.agg_method == 'mean':
            agg_out = (out_fcgnn + out_unified) / 2.0
        elif config.agg_method == 'max':
            agg_out = torch.maximum(out_fcgnn, out_unified)
        else:
            raise Exception()
        # agg_out = self.norm(agg_out)
        ffn_out = self.ffn_out(agg_out)
        # att_out, _ = self.att_aggr(agg_out)

        return ffn_out
