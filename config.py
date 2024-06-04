import json
import os
import pickle
import time
import yaml

import numpy as np
import torch
from rdkit import Chem

dataset_name = ''
config_data = ''
global_config = ''

# global
reset = ''
save_model = ''
train_log = ''
dataset_type = ''
valid_metrics = ''
root_dir = '/hy-tmp/molecule'
device = torch.device('cuda')
dataset_dir = ''

# train
train_config = ''
EPOCH = ''
BATCH_SIZE = ''
fingerprints = ''
keep_seed = ''
seed = ''
split_type = ''
random_epoch = ''
LR = ''
fcn_lr = ''
gnn_lr = ''
unified_lr = ''
task_num = ''
decay_gamma = ''
decay_step = ''
testTask = []
grad_clip = None

# net
net_config = ''
agg_method = ''
use_fingerprints_file = ''
concat_two_fingerprints = ''
use_rdkit_feature = ''
rdkit_feature_size = ''
fingerprints_catogory = ''
fingerprints_size_trans = ''
fingerprints_size_ecfp = ''
use_pretrained_atom_feature = ''
pretrained_atom_feature_size = ''
embedded_pretrained_feature_size = ''
tsfm_fp_weight_path = ''
tsfm_atom_weight_path = ''
gnn_hidden_channels = ''
gnn_out_channels = ''
gnn_num_layers = ''
fpn_hidden_channels = ''
fpn_mid_channels = ''
fpn_out_channels = ''
fcn_mid_channels = ''
fcn_out_channels = ''
unified_out_channels = ''
feature3D = ''
att_emb_dim = ''
raw_with_pos = ''
gnn_dropout = ''
unified_dropout = ''
use_random_conformer = ''
nheads = ''
aggr_type = ''
attn_outsize = ''
graph_pooling = ''
global_reducer = ''
node_reducer = ''
face_reducer = ''
use_weight_from_file = ''
mlp_hidden_size = ''
mlp_layers = ''
latent_size = ''
use_layer_norm = ''
unified_num_layers = ''
uni_checkpoint_path = ''
fingerprints_size = ''

# log 路径
cur = time.localtime(time.time())
format_day = ''
format_time = ''
format_time2 = ''
train_logpath = ''
photo_path = f'./DrawGraph/{format_day}/{format_time}'
# PNA
deg = None
# model save
model_dir = f'./checkpoint/{format_day}'
# config save
config_path = ''
data_dir = ''
tsfm_atom_fea = ''
tsfm_fp = ''
traditional_fp = ''
smi2index_file = ''
smi2index = ''
pos_3d = ''
feature_3d = ''
