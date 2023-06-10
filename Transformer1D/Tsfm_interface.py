import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

from utils.Smiles2token import smi_tokenizer, get_array, Token2Idx, NewDic, split
from pretrain_trfm import TrfmSeq2seq

model_mol = TrfmSeq2seq(len(NewDic), 256, len(NewDic), 4)
model_mol.load_state_dict(torch.load("/home/trfm_12_23000.pkl", map_location='cpu'))

model_atom = TrfmSeq2seq(len(Token2Idx), 256, len(Token2Idx), 4)
model_atom.load_state_dict(torch.load("/home/save/trfm_new_ds_lr5_47_50000.pkl", map_location='cpu'))


def Get_Atom_Feature(smi):
    """
    :param smi: 单个SMILES串
    :return: SMILES串中所有原子的特征，按照SMILES串中出现的顺序 shape = (len(atom_list), 256)
    """
    smi = smi.strip()
    tokens = smi_tokenizer(smi, max_len=None, padding=False)
    v = torch.tensor(tokens).to(torch.int)
    fea = model_atom.encode_one_all(v)
    fea = fea.squeeze()

    Mol = Chem.MolFromSmiles(smi)
    atom_list = [atom.GetSymbol() for atom in Mol.GetAtoms()]
    now = 0
    i = 0
    k = 0
    Atom_Fea = np.zeros((len(atom_list), 256), dtype=np.float32)
    while i < len(smi):
        # 匹配原子
        if now < len(atom_list):
            atom = atom_list[now]
            if len(atom) == 1 and smi[i].upper() == atom.upper():
                Atom_Fea[now] = fea[k]
                now += 1
                i += 1
                k += 1
                continue
            if len(atom) == 2 and smi[i:i + 2].upper() == atom.upper():
                Atom_Fea[now] = fea[k]
                now += 1
                i += 2
                k += 1
                continue
        # 匹配+num / -num
        if (smi[i] == '+' or smi[i] == '-') and smi[i + 1] in "1234567890":
            i += 2
            k += 1
            continue
        # 匹配 %num (编号大于等于10的环)
        if smi[i] == '%':
            i += 3
            k += 1
            continue
        # 匹配 @@ (手性异构)
        if smi[i] == '@' and smi[i + 1] == '@':
            i += 2
            k += 1
            continue
        # 匹配单个符号
        i += 1
        k += 1

    return Atom_Fea


def Get_MolFP(smiles):
    """
    :param smiles: smiles列表，长度为N
    :return: 分子特征，形状为 [N, 1024]
    """
    x_split = [split(sm.strip()) for sm in smiles]
    xid, xseg = get_array(x_split)
    X = model_mol.encode(torch.t(xid))
    return X


if __name__ == '__main__':
    smiles = pd.read_csv("/home/bace/mapping/mol.csv")['smiles']
    g = [Get_Atom_Feature(smi) for smi in tqdm(smiles)]
    # smiles = ['[Cl].CC(C)NCC(O)COc1cccc2ccccc12', 'Nc1nnc(c(N)n1)c2cccc(Cl)c2Cl', 'CCCC(C)C']
    # Get_MolFP(smiles)
