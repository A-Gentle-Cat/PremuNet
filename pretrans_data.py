import json
import os.path
import pickle

import pandas as pd
import torch
import yaml
from rdkit import Chem
from tqdm import tqdm

import config
from Utils.load_utils import load_smiles_and_label


datasets = ['BBBP', 'BACE', 'clintox', 'TOX21', 'sider', 'ESOL', 'Freesolv', 'Lipophilicity']
root = './dataset/'

def get_config(dataset):
    config.dataset_name = dataset

    with open(f'configs/{config.dataset_name}_config.yaml', 'rb') as f:
        config.config_data = yaml.unsafe_load(f)
    config.global_config = config.config_data['global']
    config.net_config = config.config_data['net']

    # global
    config.dataset_type = config.global_config['dataset_type']
    config.device = torch.device('cuda')
    config.path_dir = config.global_config['path_dir']

    config.tsfm_fp_weight_path = config.net_config['tsfm_fp_weight_path']
    config.tsfm_atom_weight_path = config.net_config['tsfm_atom_weight_path']
    config.feature3D = config.net_config['feature3D']

    if config.feature3D == 'GeoDiff':
        f = open(f'./dataset/3DFeature/{config.dataset_name}_3dfeature.pkl', 'rb')
        config.eature_3d = pickle.load(f)
    elif config.feature3D == 'GeoMol':
        f = open(f'./dataset/{config.dataset_name}/processed/{config.dataset_name}_pos3d_GeoMol_dict.pkl', 'rb')
        config.feature_3d = pickle.load(f)
    elif config.feature3D == 'GEOM':
        f = open(os.path.join(config.global_config['path_dir'], 'summary.json'), 'rb')
        # f = open('/hy-tmp/molecule_net/summary.json', 'rb')
        drugs = json.load(f)
        ori_drugs_smiles = list(drugs.keys())
        config.feature_3d = {}
        for ori_smiles in ori_drugs_smiles:
            mol = Chem.MolFromSmiles(ori_smiles)
            smiles = Chem.MolToSmiles(mol)
            config.feature_3d[smiles] = drugs[ori_smiles]['pickle_path']
    else:
        config.feature_3d = None


def start_pretrain(dataset_name):
    print(f'==================={dataset_name}===================')
    get_config(dataset_name)

    from Models.Transformer.Tsfm_interface import Get_Atom_Feature
    from FeatureDefine.Conformer3DFeature import get_3d_conformer_rdkit_ETKDG, get_3d_conformer_rdkit_MMFF, \
        get_3d_conformer_GeoMol, get_3d_conformer_GEOM, get_3d_conformer_random
    from FeatureDefine.FinerprintsFeature import getTraditionalFingerprintsFeature, getTSFMFingerprintsFeature
    data_dir = os.path.join(root, dataset_name)
    csv_path = os.path.join(data_dir, 'mapping', 'mol.csv')
    data = pd.read_csv(csv_path)

    sm2index = {}
    tsfm_atom_fea = []
    traditional_fp = []
    tsfm_fp = []
    pos_3d = []
    cnt = 0
    smiles_list = []

    for item in tqdm(data.iterrows(), desc=f'正在处理【{dataset_name}】'):
        item = item[1]
        smiles, _ = load_smiles_and_label(dataset_name, item)
        smiles_list.append(smiles)
        mol = Chem.MolFromSmiles(str(smiles))

        if config.feature3D == 'ETKDG':
            cur_pos = get_3d_conformer_rdkit_ETKDG(mol, smiles)
        elif config.feature3D == 'GeoMol':
            cur_pos = get_3d_conformer_GeoMol(mol, smiles)
        elif config.feature3D == 'MMFF':
            cur_pos = get_3d_conformer_rdkit_MMFF(mol, smiles)
        elif config.feature3D == 'GEOM':
            cur_pos = get_3d_conformer_GEOM(mol, smiles)
            if cur_pos is None:
                print('Fails to generate 3d coordinates, uses randomly distributed coordinates.', smiles)
                cur_pos = get_3d_conformer_random(mol, smiles)
        else:
            cur_pos = None

        pos_3d.append(cur_pos)

        cur_traditional_fp = getTraditionalFingerprintsFeature(mol, smiles)
        cur_tsfm_atom_fea = torch.from_numpy(Get_Atom_Feature(smiles))

        traditional_fp.append(cur_traditional_fp)
        tsfm_atom_fea.append(cur_tsfm_atom_fea)
        sm2index[smiles] = cnt
        cnt += 1
        if cur_pos is not None:
            # print(cur_tsfm_atom_fea.shape, cur_pos.shape)
            assert len(cur_pos.shape) == 3
            assert cur_tsfm_atom_fea.shape[0] == cur_pos.shape[1]
        else:
            print(f'pos is none {smiles}')

    assert cnt == len(data)

    tsfm_fp = getTSFMFingerprintsFeature(smiles_list)
    traditional_fp = torch.stack(traditional_fp)

    if not os.path.exists(f'./dataset/{dataset_name}/processed'):
        os.mkdir(f'./dataset/{dataset_name}/processed')

    # atom feature
    with open(f'./dataset/{dataset_name}/processed/{dataset_name}_tsfm_atom_fea_{config.feature3D}.pkl', 'wb') as f:
        pickle.dump(tsfm_atom_fea, f)
    # transformer fingerprint
    with open(f'./dataset/{dataset_name}/processed/{dataset_name}_tsfm_fp_{config.feature3D}.pkl', 'wb') as f:
        pickle.dump(tsfm_fp, f)
    # fingerprint
    with open(f'./dataset/{dataset_name}/processed/{dataset_name}_traditional_fp_{config.feature3D}.pkl', 'wb') as f:
        pickle.dump(traditional_fp, f)
    # index
    with open(f'./dataset/{dataset_name}/processed/{dataset_name}_smi2index_{config.feature3D}.pkl', 'wb') as f:
        pickle.dump(sm2index, f)
    # 3d position
    if config.feature3D != 'none':
        with open(f'./dataset/{dataset_name}/processed/{dataset_name}_pos3d_{config.feature3D}.pkl', 'wb') as f:
            pickle.dump(pos_3d, f)
    print(f'tsfm_fp.shape = {tsfm_fp.shape} tsfm_atom_fea.len = {len(tsfm_atom_fea)} tsfm_atom_fea[0].shape: {tsfm_atom_fea[0].shape} traditional_fea.shape = {traditional_fp.shape} pos3d.len = {len(pos_3d)} ')
    print('data save to：', f'./dataset/{dataset_name}/processed')


if __name__ == '__main__':
    # print(get_3d_conformer_GEOM('O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C').shape)
    for dataset_name in datasets:
        start_pretrain(dataset_name)
        # tmp(dataset_name)
    # for dataset_name in datasets:
    #     test(dataset_name)
    # tran_GeoMol('BACE')
