import pandas as pd
from rdkit import Chem
import numpy as np
from tqdm import tqdm


def sdf2csv():
    input_file = "/Users/rune/Downloads/chembl_27.csv"
    out_file = "/Users/rune/Downloads/chembl_27_new.csv"

    smiles = pd.read_csv(input_file)['smiles'].values

    # print(smiles)
    Mols = [Chem.MolFromSmiles(smi) for smi in smiles]

    for mol in tqdm(Mols):
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]

    smiles = [Chem.MolToSmiles(mol) for mol in Mols]

    # print(smiles)
    smiles = np.array(smiles)
    N = len(smiles)
    print(f"数据集中包含{N}个分子")
    # rands = np.random.choice(N, min(N, maxnum), replace=False)
    # smiles = smiles[rands[:]]
    df = pd.DataFrame(data=smiles, columns=['smiles'])
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    sdf2csv()
