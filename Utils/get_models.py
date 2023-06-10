import config
from FeatureDefine.MolFeature import MolFeature
from Models.GIN import GIN_Net
from Models.MV_GNN import MV_GNN_Graph
from Models.PNANet import PNA


def get_gnn_model(name, **kwargs):
    """
    @param name: model name in {GIN, MV_GNN_Graph, PNA}
    @return: gnn_model
    """
    if name == 'GIN':
        model = GIN_Net(node_channels=MolFeature.get_atom_dim(),
                        edge_channels=MolFeature.get_bond_dim(),
                        hidden_channels=config.gnn_hidden_channels,
                        out_channels=config.gnn_out_channels,
                        num_layers=config.gnn_num_layers,
                        **kwargs)
    elif name == 'PNA':
        model = PNA(in_channels=MolFeature.get_atom_dim(),
                    hidden_channels=config.gnn_hidden_channels,
                    out_channels=config.gnn_out_channels,
                    num_layers=config.gnn_num_layers,
                    aggregators=['mean', 'min', 'max', 'std'],
                    scalers=['identity', 'amplification', 'attenuation'],
                    deg=config.deg,
                    edge_dim=MolFeature().get_bond_dim(),
                    dropout=config.gnn_dropout)
    else:
        raise Exception('没有这个 2d 网络')
    return model
