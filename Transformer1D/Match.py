import json
import pickle

import molvs
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.io import loadmat
from tqdm import tqdm

# from LoadData.qm8_loader import QM8_molDataset


with open('/tmp/pycharm_project_747/summary.json', 'r') as f:
    drugs = json.load(f)
    # print(type(drugs))
    ori_drugs_smiles = list(drugs.keys())
    sider_smiles = []
    for key, value in drugs.items():
        if 'sider' in value['datasets']:
            sider_smiles.append(key)

    print(len(sider_smiles))

    # df = pd.DataFrame(data=sider_smiles, columns=['smiles'])
    # df.to_csv("sider_in_drug.csv")

    exit(0)

    for smiles in tqdm(ori_drugs_smiles, desc='正在处理 drug smiles'):
        try:
            # smiles = str(smiles).replace('[C@H]', 'C')
            # smiles = str(smiles).replace('[C@@H]', 'C')
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(mol)
            drugs_smiles.add(smiles)
        except:
            continue

exit(0)


with open('./drugs_smiles.pkl', 'wb') as f:
    pickle.dump(drugs_smiles, f)

with open('./drugs_smiles.pkl', 'rb') as f:
    drugs_smiles = pickle.load(f)

drug_fea = []
for drg_smi in drugs_smiles:
    now_fea = []
    for atom in Chem.MolFromSmiles(drg_smi).GetAtoms():
        now_fea.append((atom.GetSymbol(), 0))
    now_fea.sort()
    drug_fea.append(now_fea)

# print(drugs_smiles)

# bace = pd.read_csv('/Users/AGentleCat/PycharmProjects/molecule/dataset/BBBP/BBBP.csv')['smiles']
# bace = pd.read_csv('/Users/AGentleCat/PycharmProjects/molecule/dataset/BBBP/BACE.csv')['mol']
# bace = pd.read_csv('/Users/AGentleCat/PycharmProjects/molecule/dataset/clintox/clintox.csv')['smiles']
# bace = pd.read_csv('/Users/AGentleCat/PycharmProjects/molecule/dataset/HIV/HIV.csv')['smiles']
# bace = pd.read_csv('/Users/AGentleCat/PycharmProjects/molecule/dataset/TOX21/TOX21.csv')['smiles']
# bace = pd.read_csv('/Users/AGentleCat/PycharmProjects/molecule/dataset/TOXCAST/TOXCAST.csv')['smiles']
bace = pd.read_csv('/tmp/pycharm_project_747/dataset/sider.csv')['smiles']
bace_smiles = [bace[i] for i in range(len(bace))]
matched = 0
dismatched = 0

for smiles in tqdm(bace_smiles, desc='正在匹配 数据集smiles'):
    try:
        # smiles = str(smiles).replace('[C@H]', 'C')
        # smiles = str(smiles).replace('[C@@H]', 'C')
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol)
    except:
        print("?")
        continue

    flag = False

    # 得到smiles的特征
    atm_fea = []
    for atom in Chem.MolFromSmiles(smiles).GetAtoms():
        atm_fea.append((atom.GetSymbol(), 0))

    atm_fea.sort()

    if atm_fea in drug_fea:
        flag = True

    if flag:
        matched += 1
    else:
        dismatched += 1

print(f'matched: {matched} dismatched: {dismatched}')
