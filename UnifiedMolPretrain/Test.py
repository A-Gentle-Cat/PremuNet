from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit import Chem

from pretrain3d.data.pcqm4m_lazy import PCQM4Mv2LasyDataset
from pretrain3d.utils.graph import smiles2graphwithface
from tqdm import tqdm
from pretrain3d.utils.features import get_atom_feature_dims

dataset = PCQM4Mv2LasyDataset()
print(dataset[0])
