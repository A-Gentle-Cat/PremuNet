import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from FeatureDefine.pubchemfp import GetPubChemFPs

def getTraditionalFingerprintsFeature(mol: Chem.Mol, smiles=None):
    fp2 = []
    fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
    fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    fp_pubcfp = GetPubChemFPs(mol)
    # print(f'maccs: {torch.tensor(fp_maccs).shape} pubchem: {torch.tensor(fp_pubcfp).shape} phaerg: {torch.tensor(fp_phaErGfp).shape}')
    fp2.extend(fp_maccs)
    fp2.extend(fp_phaErGfp)
    fp2.extend(fp_pubcfp)
    fp2 = torch.tensor(fp2, dtype=torch.float32)

    return fp2


def getTSFMFingerprintsFeature(smiles_list):
    from Models.Transformer.Tsfm_interface import Get_MolFP
    # fp = []
    # for smiles in smiles_list:
    #     fp.append(torch.tensor(smi_fingerprint(smiles), dtype=torch.float32).to('cpu'))
    # fp = torch.stack(fp)
    # return fp
    return torch.tensor(Get_MolFP(smiles_list), dtype=torch.float32).to('cpu')
