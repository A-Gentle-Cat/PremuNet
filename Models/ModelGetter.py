import torch

from FeatureDefine.MolFeature import MolFeature
from Models.AblationModel import FPN_Tradi
from Models.BasicGNN import GIN
from Models.FPN import FPN, SMILES_Transformer
from Models.FP_GNN_NET import FP_GNN_NET
from Models.GIN import GIN_Net
from Models.Layers.comenet import ComENet
from Models.Layers.dimenetpp import DimeNetPP
from Models.Layers.schnet import SchNet
from Models.Layers.spherenet import SphereNet
from Models.PNANet import PNA
from Models.Sphere_Net import Sch_Net, Sphere_Net
from Models.TSFM import Tsfm_class
from Models.Uni_MPNN import Uni_Net
from UnifiedMolPretrain.pretrain3d.model.gnn import GNNet
import config


def get_model(model_name):
    model = None
    if model_name == 'PremuNet':
        model = Uni_Net()
    if model_name == 'PremuNet-L':
        model = FP_GNN_NET(predictor=True)
    if model_name == 'PremuNet-H':
        if config.dataset_type == 'classification':
            model = GNNet(num_message_passing_steps=12,
                          mlp_hidden_size=1024,
                          latent_size=256,
                          mlp_layers=2,
                          node_attn=True,
                          use_bn=True,
                          graph_pooling=config.graph_pooling,
                          global_reducer=config.global_reducer,
                          node_reducer=config.node_reducer,
                          dropout=0.0,
                          dropedge_rate=0.0,
                          dropnode_rate=0.0,
                          num_tasks=config.task_num,
                          raw_with_pos=config.raw_with_pos)
        else:
            model = GNNet(num_message_passing_steps=12,
                          mlp_hidden_size=1024,
                          latent_size=256,
                          mlp_layers=2,
                          node_attn=True,
                          use_bn=True,
                          graph_pooling=config.graph_pooling,
                          global_reducer=config.global_reducer,
                          node_reducer=config.node_reducer,
                          num_tasks=config.task_num,
                          raw_with_pos=config.raw_with_pos)
        if config.use_weight_from_file:
            checkpoint = torch.load(config.uni_checkpoint_path, map_location=torch.device("cpu"))[
                "model_state_dict"
            ]
            model.load_state_dict(checkpoint)
    if model_name == 'FPN':
        model = FPN(in_channals=config.fingerprints_size,
                    hidden_channals=config.fpn_hidden_channels,
                    mid_channals=config.fpn_mid_channels,
                    out_channals=config.task_num,
                    drop_p=config.gnn_dropout)
    if model_name == 'SMILES-Transformer':
        model = SMILES_Transformer(in_channals=config.fingerprints_size_trans,
                                   hidden_channals=config.fpn_hidden_channels,
                                   mid_channals=config.fpn_mid_channels,
                                   out_channals=config.task_num,
                                   drop_p=config.gnn_dropout)
    if model_name == 'PNA':
        model = PNA(in_channels=MolFeature.get_atom_dim(),
                    hidden_channels=config.gnn_hidden_channels,
                    out_channels=config.task_num,
                    num_layers=config.gnn_num_layers,
                    aggregators=['mean', 'min', 'max', 'std'],
                    scalers=['identity', 'amplification', 'attenuation'],
                    deg=config.deg,
                    edge_dim=MolFeature().get_bond_dim(),
                    dropout=config.gnn_dropout,
                    pnanet=True)
    if model_name == 'PNA-FP':
        model = PNA(in_channels=MolFeature.get_atom_dim(),
                    hidden_channels=config.gnn_hidden_channels,
                    out_channels=config.task_num,
                    num_layers=config.gnn_num_layers,
                    aggregators=['mean', 'min', 'max', 'std'],
                    scalers=['identity', 'amplification', 'attenuation'],
                    deg=config.deg,
                    edge_dim=MolFeature().get_bond_dim(),
                    dropout=config.gnn_dropout,
                    jk='lstm',
                    pnanet=False)
    if model_name == 'GIN':
        model = GIN(in_channels=MolFeature().get_atom_dim(),
                    hidden_channels=config.gnn_hidden_channels,
                    num_layers=config.gnn_num_layers,
                    out_channels=config.task_num)
    if model_name == 'SchNet':
        model = SchNet(out_channels=config.task_num)
    if model_name == 'SphereNet':
        model = SphereNet(hidden_channels=128,
                          int_emb_size=64,
                          out_channels=config.task_num)
    if model_name == 'DimeNet':
        model = DimeNetPP(out_channels=config.task_num)
    if model_name == 'TSFM':
        model = Tsfm_class(hidden_size=config.hidden_size,
                           n_layers=3,
                           dic_len=100,
                           task=config.task_num,
                           l_hid=config.hidden_size)
    if model_name == 'Tradi-FPN':
        model = FPN_Tradi(in_channals=config.fingerprints_size_ecfp,
                          hidden_channals=config.fpn_hidden_channels,
                          mid_channals=config.fpn_mid_channels,
                          out_channals=config.task_num,
                          drop_p=config.gnn_dropout)
    if model_name == 'ComENet':
        model = ComENet(out_channels=config.task_num)
    return model
