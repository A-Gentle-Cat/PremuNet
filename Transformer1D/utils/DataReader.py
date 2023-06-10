import pandas as pd
import numpy as np
from rdkit import Chem
# import chembl_downloader


def BBBP_reader():
    file_path = '/Users/rune/PycharmProjects/Transformer1D/dataset/BBBP.csv'
    Data = pd.read_csv(file_path)
    labels = Data["p_np"].values
    smiles = Data["smiles"].values
    return smiles, labels


def Chem_reader():
    file_path = "/Users/rune/PycharmProjects/Transformer1D/dataset/chembl_v27_standardized.sdf"
    Mols = Chem.SDMolSupplier(file_path)
    smiles = [Chem.MolToSmiles(mol) for mol in Mols]
    print(smiles)


def download_chembl():
    path = chembl_downloader.download_sdf(version='24')
    print(path)


if __name__ == '__main__':
    download_chembl()
