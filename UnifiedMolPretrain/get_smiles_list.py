import pickle

from rdkit import Chem
from rdkit.Chem import SDMolSupplier
from tqdm import tqdm

sdf_file = "./dataset/pcqm4m-v2-train.sdf"
sup = SDMolSupplier(sdf_file)
print("读取sdf完成")
print('开始生成smiles串')
smiles_list = [Chem.MolToSmiles(mol) for mol in tqdm(sup, desc='正在处理 smiles 串')]
with open('./dataset/smiles_list.pkl', 'wb') as f:
    pickle.dump(smiles_list, f)
print('已生成smiles串')

with open('./dataset/smiles_list.pkl', 'rb') as f:
    smiles_list = pickle.load(f)
print(len(smiles_list))
print(smiles_list[0])

