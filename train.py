import argparse
import json
import os
import pickle
import time

import torch
import yaml
from rdkit import Chem


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='', choices=['BBBP', 'BACE', 'clintox', 'sider', 'TOX21', 'ESOL', 'Freesolv', 'HIV', 'Lipophilicity'], required=True)
    parser.add_argument('--reset', default=None, action='store_true', help='whether to regenerate the processed file')
    parser.add_argument('--keep_seed', default=None, action='store_true', help='automatically generate random seed at each fold, if false, use --seed')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epoch', type=int ,default=None)
    parser.add_argument('--random_epoch', type=int, default=None, help='number of folds for evaluating model')
    parser.add_argument('--model', type=str, default='PremuNet', choices=['PremuNet', 'PremuNet-L', 'PremuNet-H', 'FPN', 'GIN', 'PNA', 'SchNet', 'SphereNet', 'SMILES-Transformer', 'DimeNet', 'Tradi-FPN', 'PNA-FP', 'ComENet'])
    parser.add_argument('--print_to_log', default=False, action='store_true', help='whether transfer console output to the log file')
    parser.add_argument('--agg_method', default='concat', choices=['concat', 'add', 'mean', 'sum', 'max'])
    parser.add_argument('--LR', type=float, default=None)
    parser.add_argument('--decay_gamma', type=float, default=None)
    parser.add_argument('--decay_step', type=float, default=None)

    args = parser.parse_args()
    return args

def get_config(args):
    config.dataset_name = args.dataset

    with open(f'configs/{config.dataset_name}_config.yaml', 'rb') as f:
        config.config_data = yaml.unsafe_load(f)
    config.global_config = config.config_data['global']

    # global
    config.reset = args.reset if args.reset is not None else config.global_config['reset']
    config.save_model = config.global_config['save_model']
    config.train_log = config.global_config['train_log']
    config.dataset_type = config.global_config['dataset_type']
    config.valid_metrics = config.global_config['valid_func']
    config.root_dir = '/hy-tmp/molecule'
    config.device = torch.device('cuda')
    config.path_dir = config.global_config['path_dir']

    # train
    config.train_config = config.config_data['train']
    config.EPOCH = args.epoch if args.epoch is not None else config.train_config['EPOCH']
    config.BATCH_SIZE = config.train_config['BATCH_SIZE']
    config.fingerprints = config.train_config['fingerprints']
    config.keep_seed = args.keep_seed if args.keep_seed is not None else config.train_config['keep_seed']
    config.seed = args.seed if args.seed is not None else config.train_config['seed']
    config.split_type = config.train_config['split_type']
    config.random_epoch = args.random_epoch if args.random_epoch is not None else config.train_config['random_epoch']
    config.LR = args.LR if args.LR is not None else float(config.train_config['LR'])
    config.fcn_lr = float(config.train_config['fcn_lr'])
    config.gnn_lr = float(config.train_config['gnn_lr'])
    config.unified_lr = float(config.train_config['unified_lr'])
    config.task_num = config.train_config['task_num']
    config.decay_gamma = args.decay_gamma if args.decay_gamma is not None else config.train_config['decay_gamma']
    config.decay_step = args.decay_step if args.decay_step is not None else config.train_config['decay_step']
    config.testTask = []
    config.grad_clip = config.train_config['grad_clip']

    # net
    config.net_config = config.config_data['net']

    config.agg_method = args.agg_method
    config.net_config['agg_method'] = args.agg_method

    config.use_fingerprints_file = config.net_config['use_fingerprints_file']
    config.concat_two_fingerprints = config.net_config['concat_two_fingerprints']
    config.use_rdkit_feature = config.net_config['use_rdkit_feature']
    config.rdkit_feature_size = config.net_config['rdkit_feature_size']
    config.fingerprints_catogory = config.net_config['fingerprints_catogory']
    config.fingerprints_size_trans = config.net_config['fingerprints_size_trans']
    config.fingerprints_size_ecfp = config.net_config['fingerprints_size_ecfp']
    config.use_pretrained_atom_feature = config.net_config['use_pretrained_atom_feature']
    config.pretrained_atom_feature_size = config.net_config['pretrained_atom_feature_size']
    config.embedded_pretrained_feature_size = config.net_config['embedded_pretrained_feature_size']
    config.tsfm_fp_weight_path = config.net_config['tsfm_fp_weight_path']
    config.tsfm_atom_weight_path = config.net_config['tsfm_atom_weight_path']
    config.feature3D = config.net_config['feature3D']
    config.att_emb_dim = config.net_config['att_emb_dim']
    config.raw_with_pos = config.net_config['raw_with_pos']
    config.gnn_dropout = float(config.net_config['gnn_dropout'])
    config.unified_dropout = float(config.net_config['unified_dropout'])
    config.gnn_hidden_channels = config.net_config['gnn_hidden_channels']
    config.gnn_out_channels = config.net_config['gnn_out_channels']
    config.gnn_num_layers = config.net_config['gnn_num_layers']
    config.fpn_hidden_channels = config.net_config['fpn_hidden_channels']
    config.gnn_hidden_channels = config.net_config['gnn_hidden_channels']
    config.fpn_mid_channels = config.net_config['fpn_mid_channels']
    config.fpn_out_channels = config.net_config['fpn_out_channels']
    config.fcn_mid_channels = config.net_config['fcn_mid_channels']
    config.fcn_out_channels = config.net_config['fcn_out_channels']
    config.unified_out_channels = config.net_config['unified_out_channels']
    config.use_random_conformer = bool(config.net_config['use_random_conformer'])
    config.nheads = config.net_config['nheads']
    config.aggr_type = config.net_config['aggr_type']
    config.attn_outsize = config.net_config['attn_outsize']
    config.graph_pooling = config.net_config['graph_pooling']
    config.global_reducer = config.net_config['global_reducer']
    config.node_reducer = config.net_config['node_reducer']
    config.face_reducer = config.net_config['face_reducer']
    config.use_weight_from_file = config.net_config['use_weight_from_file']
    config.mlp_hidden_size = config.net_config['mlp_hidden_size']
    config.mlp_layers = config.net_config['mlp_layers']
    config.latent_size = config.net_config['latent_size']
    config.use_layer_norm = config.net_config['use_layer_norm']
    config.unified_num_layers = config.net_config['unified_num_layers']
    config.uni_checkpoint_path = config.net_config['uni_checkpoint_path']

    if config.concat_two_fingerprints:
        config.fingerprints_size = config.fingerprints_size_ecfp + config.fingerprints_size_trans
    elif config.fingerprints_catogory == 'ecfp':
        config.fingerprints_size = config.fingerprints_size_ecfp
    else:
        config.fingerprints_size = config.fingerprints_size_trans

    # log 路径
    config.cur = time.localtime(time.time())
    config.format_day = '%04d-%02d-%02d' % (config.cur.tm_year, config.cur.tm_mon, config.cur.tm_mday)
    config.format_time = '【%04d-%02d-%02d】%02d:%02d:%02d' % (
    config.cur.tm_year, config.cur.tm_mon, config.cur.tm_mday, config.cur.tm_hour, config.cur.tm_min, config.cur.tm_sec)
    config.format_time2 = '%02d:%02d:%02d' % (config.cur.tm_hour, config.cur.tm_min, config.cur.tm_sec)
    print(os.getcwd())
    if config.train_log:
        if not os.path.exists(f'./TrainLogs/{config.format_day}'):
            os.mkdir(f'./TrainLogs/{config.format_day}')
        config.train_logpath = f'./TrainLogs/{config.format_day}/{config.format_time}_{config.dataset_name}.log'
    else:
        config.train_logpath = None

    if not os.path.exists(f'./DrawGraph/{config.format_day}'):
        os.mkdir(f'./DrawGraph/{config.format_day}')
    config.photo_path = f'./DrawGraph/{config.format_day}/{config.format_time}'

    # PNA
    config.deg = None

    # 模型保存
    if not os.path.exists(f'./checkpoint/{config.format_day}'):
        os.mkdir(f'./checkpoint/{config.format_day}')
    config.model_dir = f'./checkpoint/{config.format_day}'

    # config 保存
    if not os.path.exists(f'./TrainConfigs/{config.format_day}'):
        os.mkdir(f'./TrainConfigs/{config.format_day}')
    config.config_path = f'./TrainConfigs/{config.format_day}/{config.format_time}.yaml'
    with open(config.config_path, 'w') as f:
        yaml.dump(config.config_data, f)

    # 特征文件读入
    os.chdir(config.root_dir)
    config.data_dir = f'./dataset/{config.dataset_name}/processed/'
    config.tsfm_atom_fea = f'{config.dataset_name}_tsfm_atom_fea_{config.feature3D}.pkl'
    config.tsfm_fp = f'{config.dataset_name}_tsfm_fp_{config.feature3D}.pkl'
    config.traditional_fp = f'{config.dataset_name}_traditional_fp_{config.feature3D}.pkl'
    config.smi2index_file = f'{config.dataset_name}_smi2index_{config.feature3D}.pkl'
    print(config.tsfm_fp, config.tsfm_atom_fea)

    if not (os.path.exists(os.path.join(config.data_dir, config.tsfm_atom_fea)) and os.path.exists(
            os.path.join(config.data_dir, config.tsfm_fp)) and os.path.exists(
            os.path.join(config.data_dir, config.traditional_fp)) and os.path.join(
            os.path.join(config.data_dir, config.smi2index_file))):
        print('警告：没有找到特征文件，需要先运行 pretrans_data.py 获取特征')
    else:
        with open(os.path.join(config.data_dir, config.tsfm_fp), 'rb') as f:
            config.tsfm_fp = pickle.load(f)
        with open(os.path.join(config.data_dir, config.traditional_fp), 'rb') as f:
            config.traditional_fp = pickle.load(f)
        with open(os.path.join(config.data_dir, config.tsfm_atom_fea), 'rb') as f:
            config.tsfm_atom_fea = pickle.load(f)
        with open(os.path.join(config.data_dir, config.smi2index_file), 'rb') as f:
            config.smi2index = pickle.load(f)
        with open(f'./dataset/{config.dataset_name}/processed/{config.dataset_name}_pos3d_{config.feature3D}.pkl', 'rb') as f:
            config.pos_3d = pickle.load(f)

    if config.feature3D == 'GeoDiff':
        f = open(f'./dataset/3DFeature/{config.dataset_name}_3dfeature.pkl', 'rb')
        config.eature_3d = pickle.load(f)
    elif config.feature3D == 'GeoMol':
        f = open(f'./dataset/{config.dataset_name}/processed/{config.dataset_name}_pos3d_GeoMol_dict.pkl', 'rb')
        config.feature_3d = pickle.load(f)
    elif config.feature3D == 'GEOM':
        f = open('/hy-tmp/molecule_net/summary.json', 'rb')
        drugs = json.load(f)
        ori_drugs_smiles = list(drugs.keys())
        config.feature_3d = {}
        for ori_smiles in ori_drugs_smiles:
            mol = Chem.MolFromSmiles(ori_smiles)
            smiles = Chem.MolToSmiles(mol)
            config.feature_3d[smiles] = drugs[ori_smiles]['pickle_path']
    else:
        config.feature_3d = None

args = parse_arguments()

import config
get_config(args)

if args.print_to_log:
    import sys

    if not os.path.exists(f'./log/{config.format_day}'):
        os.makedirs(f'./log/{config.format_day}')
    f = open(f'./log/{config.format_day}/{config.format_time}_{args.dataset}.log', 'a')
    sys.stdout = f
    sys.stderr = f

with open(f'configs/{args.dataset}_config.yaml', 'rb') as f:
    new_config_data = yaml.unsafe_load(f)
new_config_data['global']['dataset_name'] = args.dataset
new_config_data['global']['reset'] = args.reset if args.reset is not None else new_config_data['global']['reset']
new_config_data['global']['keep_seed'] = args.keep_seed if args.keep_seed is not None else new_config_data['train']['keep_seed']
new_config_data['global']['seed'] = args.seed if args.reset is not None else new_config_data['train']['seed']
new_config_data['global']['random_epoch'] = args.random_epoch if args.random_epoch is not None else new_config_data['train']['random_epoch']

if not os.path.exists(f'./TrainConfigs/{config.format_day}'):
    os.mkdir(f'./TrainConfigs/{config.format_day}')
config.config_path = f'./TrainConfigs/{config.format_day}/{config.format_time}.yaml'

with open(config.config_path, 'w') as f:
    yaml.dump(new_config_data, f)

if __name__ == '__main__':
    from run_training import start_split_train # do not move
    start_split_train(args)


