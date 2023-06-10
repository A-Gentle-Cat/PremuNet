from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit import Chem
from pretrain3d.utils.graph import smiles2graphwithface
from tqdm import tqdm

sdf_file = "/root/pcqm4m-v2-train.sdf"

sup = SDMolSupplier(sdf_file)

for i, mol in tqdm(enumerate(sup)):
    fea = smiles2graphwithface(mol)

# smi = "COP(=O)(OC)/C(F)=C(\C)F"
#
# fea = smiles2graphwithface(smi)