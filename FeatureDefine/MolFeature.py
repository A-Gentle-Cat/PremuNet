import numpy as np
import torch
from rdkit import Chem
import torch.nn.functional as F

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


class MolFeature():
    allowable_features = {
        "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
        "possible_chirality_list": [
            "CHI_UNSPECIFIED",
            "CHI_TETRAHEDRAL_CW",
            "CHI_TETRAHEDRAL_CCW",
            "CHI_OTHER",
        ],
        "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
        "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
        "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
        "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
        "possible_is_aromatic_list": [False, True],
        "possible_is_in_ring_list": [False, True],
        "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
        "possible_bond_stereo_list": [
            "STEREONONE",
            "STEREOZ",
            "STEREOE",
            "STEREOCIS",
            "STEREOTRANS",
            "STEREOANY",
        ],
        "possible_is_conjugated_list": [False, True],
    }

    def __init__(self):
        pass

    @staticmethod
    def get_node_feature(mol: Chem.Mol):
        feature = []
        for atom in mol.GetAtoms():
            feature.append(MolFeature.atom_to_feature_vector(atom))
        feature = torch.tensor(np.array(feature), dtype=torch.int64)
        feature = MolFeature.one_hot_atoms(feature)
        return feature


    @staticmethod
    def get_bond_feature(mol: Chem.Mol):
        edge_index = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = MolFeature.bond_to_feature_vector(bond)
            edge_index.append((i, j))
            edge_features_list.append(edge_feature)
            edge_index.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.int64)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.int64)
        # print(edge_attr.shape, Chem.MolToSmiles(mol))
        edge_attr = MolFeature.one_hot_bonds(edge_attr)

        return edge_index, edge_attr

    @staticmethod
    def get_atoms_type(mol: Chem.Mol):
        atom_types = []
        for atom in mol.GetAtoms():
            atom_types.append(atom.GetAtomicNum())
        return torch.tensor(np.array(atom_types))

    @staticmethod
    def atom_to_feature_vector(atom: Chem.Atom):
        """
        Converts rdkit atom object to feature list of indices
        :param mol: rdkit atom object
        :return: list
        """
        atom_feature = [
            safe_index(MolFeature.allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
            MolFeature.allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
            safe_index(MolFeature.allowable_features["possible_degree_list"], atom.GetTotalDegree()),
            safe_index(MolFeature.allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()),
            safe_index(MolFeature.allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
            safe_index(
                MolFeature.allowable_features["possible_number_radical_e_list"], atom.GetNumRadicalElectrons()
            ),
            safe_index(MolFeature.allowable_features["possible_hybridization_list"], str(atom.GetHybridization())),
            MolFeature.allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
            MolFeature.allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
        ]
        return atom_feature

    @staticmethod
    def get_atom_feature_dims_list():
        return list(
            map(
                len,
                [
                    MolFeature.allowable_features["possible_atomic_num_list"],
                    MolFeature.allowable_features["possible_chirality_list"],
                    MolFeature.allowable_features["possible_degree_list"],
                    MolFeature.allowable_features["possible_formal_charge_list"],
                    MolFeature.allowable_features["possible_numH_list"],
                    MolFeature.allowable_features["possible_number_radical_e_list"],
                    MolFeature.allowable_features["possible_hybridization_list"],
                    MolFeature.allowable_features["possible_is_aromatic_list"],
                    MolFeature.allowable_features["possible_is_in_ring_list"],
                ],
            )
        )

    @staticmethod
    def get_atom_dim():
        return sum(MolFeature.get_atom_feature_dims_list())

    @staticmethod
    def get_bond_dim():
        return sum(MolFeature.get_bond_feature_dims_list())

    @staticmethod
    def bond_to_feature_vector(bond):
        """
        Converts rdkit bond object to feature list of indices
        :param mol: rdkit bond object
        :return: list
        """
        bond_feature = [
            safe_index(MolFeature.allowable_features["possible_bond_type_list"], str(bond.GetBondType())),
            MolFeature.allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
            MolFeature.allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
        ]
        return bond_feature

    @staticmethod
    def get_bond_feature_dims_list():
        return list(
            map(len,[MolFeature.allowable_features["possible_bond_type_list"],
                     MolFeature.allowable_features["possible_bond_stereo_list"],
                     MolFeature.allowable_features["possible_is_conjugated_list"]],)
        )

    @staticmethod
    def atom_feature_vector_to_dict(atom_feature):
        [
            atomic_num_idx,
            chirality_idx,
            degree_idx,
            formal_charge_idx,
            num_h_idx,
            number_radical_e_idx,
            hybridization_idx,
            is_aromatic_idx,
            is_in_ring_idx,
        ] = atom_feature

        feature_dict = {
            "atomic_num": MolFeature.allowable_features["possible_atomic_num_list"][atomic_num_idx],
            "chirality": MolFeature.allowable_features["possible_chirality_list"][chirality_idx],
            "degree": MolFeature.allowable_features["possible_degree_list"][degree_idx],
            "formal_charge": MolFeature.allowable_features["possible_formal_charge_list"][formal_charge_idx],
            "num_h": MolFeature.allowable_features["possible_numH_list"][num_h_idx],
            "num_rad_e": MolFeature.allowable_features["possible_number_radical_e_list"][number_radical_e_idx],
            "hybridization": MolFeature.allowable_features["possible_hybridization_list"][hybridization_idx],
            "is_aromatic": MolFeature.allowable_features["possible_is_aromatic_list"][is_aromatic_idx],
            "is_in_ring": MolFeature.allowable_features["possible_is_in_ring_list"][is_in_ring_idx],
        }

        return feature_dict

    @staticmethod
    def bond_feature_vector_to_dict(bond_feature):
        [bond_type_idx, bond_stereo_idx, is_conjugated_idx] = bond_feature
        feature_dict = {
            "bond_type": MolFeature.allowable_features["possible_bond_type_list"][bond_type_idx],
            "bond_stereo": MolFeature.allowable_features["possible_bond_stereo_list"][bond_stereo_idx],
            "is_conjugated": MolFeature.allowable_features["possible_is_conjugated_list"][is_conjugated_idx],
        }
        return feature_dict

    @staticmethod
    def one_hot_bonds(bonds: torch.Tensor):
        vocab_sizes = MolFeature.get_bond_feature_dims_list()
        one_hots = []
        bonds = bonds.reshape((-1, len(MolFeature.get_bond_feature_dims_list())))
        for i in range(bonds.shape[1]):
            one_hots.append(F.one_hot(bonds[:, i], num_classes=vocab_sizes[i]).to(bonds.device))
        return torch.cat(one_hots, dim=1).to(torch.float32)

    @staticmethod
    def one_hot_atoms(atoms: torch.Tensor) -> torch.Tensor:
        vocab_sizes = MolFeature.get_atom_feature_dims_list()
        one_hots = []
        atoms = atoms.reshape((-1, len(MolFeature.get_atom_feature_dims_list())))
        for i in range(atoms.shape[1]):
            one_hots.append(F.one_hot(atoms[:, i], num_classes=vocab_sizes[i]).to(atoms.device))

        return torch.cat(one_hots, dim=1).to(torch.float32)
