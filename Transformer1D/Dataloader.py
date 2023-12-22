import random
import pandas as pd
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
# import molvs.standardize
from rdkit import Chem
from rdkit.Chem import Draw

from SmilesEnumerator import SmilesEnumerator
from utils.Smiles2token import smi_tokenizer


class SeqDataset(Dataset):

    def __init__(self, smiles, max_len, mask_prob):
        self.smiles = smiles
        self.sme = SmilesEnumerator()
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        smi = self.smiles[item]
        try:
            smi = self.sme.randomize_smiles(smi)
        except Exception as e:
            print(e)
            print(smi, "无法random")
        X = smi_tokenizer(smi, max_len=self.max_len, padding=True)
        masked_x = smi_tokenizer(smi, max_len=self.max_len, padding=True, mask_prob=self.mask_prob)
        return torch.tensor(masked_x), torch.tensor(X)


if __name__ == '__main__':
    # F = open(r'/home/smiles_list.pkl', 'rb')
    # smiles = pickle.load(F)
    smiles = pd.read_csv("/home/bbbp/mapping/mol.csv")['smiles'].values
    smi_len = [len(smi) for smi in smiles]
    print(max(smi_len))
