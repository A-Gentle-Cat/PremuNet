import os.path
import pickle
from imp import reload
from typing import Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import config

def get_3d_conformer_rdkit_ETKDG(mol: Chem.Mol, smiles, use_file=False):
    if len(smiles) >= 200:
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=42, useRandomCoords=True)
    else:
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=42)
    conformers = mol.GetConformers()
    if len(conformers) == 0:
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=42, useRandomCoords=True)
    conformers = mol.GetConformers()
    if len(conformers) == 0:
        return None
    conformer = conformers[0]
    point = torch.tensor(np.array(conformer.GetPositions()), dtype=torch.float32)
    return point

def get_3d_conformer_rdkit_MMFF(mol: Chem.Mol, smiles):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=np.random.randint(0, 10000), maxAttempts=5000)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
        # print(mol)
        conformers = mol.GetConformers()
        if len(conformers) == 0:
            return None
        all_pos = []
        for conformer in conformers:
            pos = conformer.GetPositions()
            all_pos.append(pos)
        all_pos = torch.from_numpy(np.array(all_pos))
    except:
        return None
    return all_pos

def get_3d_conformer_GeoMol(mol: Chem.Mol, smiles):
    con_list = config.feature_3d.get(smiles)
    if con_list is None:
        return None
    point = []
    for mol in con_list:
        conformer_seq = mol.GetConformers()
        conformer = conformer_seq[0]
        point.append(conformer.GetPositions())
    point = torch.tensor(np.array(point[0]), dtype=torch.float32)
    # print(point.shape)
    return point

def get_3d_conformer_Geodiff(smiles):
    return config.feature_3d.get(smiles)

def get_3d_conformer_GEOM(mol: Chem.Mol, ori_smiles) -> Union[torch.Tensor, None]:
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(ori_smiles))
    pic_path = config.feature_3d.get(smiles)
    if pic_path is None:
        return None
    path = os.path.join(config.path_dir, pic_path)
    with open(path, 'rb') as f:
        mols = pickle.load(f)
    positions = []
    # print('conformers', len(mols['conformers']))
    las_atomic_numbers = []

    for conformer in mols['conformers']:
        # 检查顺序
        # atomic_numbers = np.array([atom.GetAtomicNum() for atom in conformer['rd_mol'].GetAtoms()])
        # for las_list in las_atomic_numbers:
        #     if not all(las_list == atomic_numbers):
        #         raise Exception()
        # las_atomic_numbers.append(atomic_numbers)

        pos = conformer['rd_mol'].GetConformers()[0].GetPositions()
        positions.append(pos)
    return torch.tensor(np.array(positions))

def get_3d_conformer_random(mol: Chem.Mol, smiles):
    mol = Chem.AddHs(mol)
    smiles = Chem.MolToSmiles(mol)
    pos = torch.zeros((1, mol.GetNumAtoms(), 3)).uniform_(-1, 1)
    return pos


# def get_pretrans_feature()
if __name__ == '__main__':
    smiles = 'C1C[C@@H]2[C@@H]3CC[C@H](C3)[C@@H]2C1'
    mol = Chem.MolFromSmiles(smiles)
    print(get_3d_conformer_random(mol, smiles).shape)
