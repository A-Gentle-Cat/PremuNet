from multiprocessing import Pool

import numpy as np
import torch
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from rdkit import Chem
from tqdm import tqdm


def rdkit_2d_features_generator(mol: Chem.Mol) -> torch.Tensor:
    """
    Generates RDKit 2D features for a molecule.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :return: A 1D Tensor containing the RDKit 2D features.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]

    return torch.tensor(features, dtype=torch.float32)

# def rdkit_2d_features_generator(data):
#     mols = (d.smiles for d in data)
#     features_generator = get_features_generator('rdkit_2d')
#     features_map = Pool(30).imap(features_generator, mols)
#
#     # Get features
#     temp_features = []
#     for i, feats in tqdm(enumerate(features_map), total=len(data)):
#         temp_features.append(feats)
#
#     return torch.tensor(temp_features)