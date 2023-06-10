import json
import os
import pickle

import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm

from FeatureDefine.FinerprintsFeature import getTSFMFingerprintsFeature
from Utils.load_utils import load_smiles_and_label

root = './dataset/'
counter = {}
f = open('/hy-tmp/molecule_net/summary.json', 'rb')
drugs = json.load(f)
ori_drugs_smiles = list(drugs.keys())
feature_3d = {}
for ori_smiles in ori_drugs_smiles:
    mol = Chem.MolFromSmiles(ori_smiles)
    smiles = Chem.MolToSmiles(mol)
    feature_3d[smiles] = drugs[ori_smiles]['pickle_path']

def start(dataset_name):
    data_dir = os.path.join(root, dataset_name)
    csv_path = os.path.join(data_dir, 'mapping', 'mol.csv')
    data = pd.read_csv(csv_path)

    count_all = 0
    count_match = 0

    for item in tqdm(data.iterrows(), desc=f'正在处理【{dataset_name}】'):
        item = item[1]
        smiles, label = load_smiles_and_label(dataset_name, item)
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        if feature_3d.get(smiles) is not None:
            count_match += 1
        count_all += 1

    return {"matched": count_match, "all": count_all, "rate": count_match / count_all}




if __name__ == '__main__':
    dataset_list = ['BBBP', 'BACE', 'clintox', 'TOX21', 'sider', 'ESOL', 'Freesolv', 'Lipophilicity']
    # dataset_list = ['Freesolv', 'Lipophilicity']
    for dataset_name in dataset_list:
        counter[dataset_name] = start(dataset_name)
    print(counter)
