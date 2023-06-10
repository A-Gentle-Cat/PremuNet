from typing import Optional

import pandas as pd
from tqdm import tqdm
import torch_geometric as pyg

from torch_geometric.data import (
    Data,
    Dataset
)

from FeatureDefine.AtomFeature import AtomFeature
from FeatureDefine.BondFeature import BondFeature
from config import *

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

if use_feature_file and dataset_name == 'QM9':
    data_dir = f'./dataset/{dataset_name}/processed/'

    tsfm_file = f'{dataset_name}_tsfm_fea.pkl'
    smi2index_file = f'{dataset_name}_smi2index.pkl'
    with open(os.path.join(data_dir, tsfm_file), 'rb') as f:
        tsfm_fea = pickle.load(f)
    with open(os.path.join(data_dir, smi2index_file), 'rb') as f:
        smi2index = pickle.load(f)

    data_ev = np.load('./dataset/QM9/raw/qm9_eV.npz')
    N = data_ev['N']
    R = data_ev['R']
    ids = data_ev['id']
    split = np.cumsum(N)
    pos_3d = np.split(R, split)
    pos_3d = {f'gdb_{id}': pos for id, pos in zip(ids, pos_3d)}
    print('pos_3d.len =', len(pos_3d))
    smiles_fp = torch.load(f'./dataset/{dataset_name}/processed/{dataset_name}_smiles_fp.pt')


class QM9_molDataset(Dataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - #graphs
              - #nodes
              - #edges
              - #features
              - #tasks
            * - 130,831
              - ~18.0
              - ~37.3
              - 11
              - 19
    """  # noqa: E501

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'
    task_names =  ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0',
                   'u298', 'h298', 'g298', 'cv','u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']

    def __init__(self):
        self.root = './dataset/QM9'
        super().__init__(self.root)
        if config.reset or not os.path.exists(self.processed_paths[0]):
            self.process()
        self.data_list = torch.load(self.processed_paths[0])
        self.data_length = len(self.data_list)
        print(f'读入预处理文件成功! 总长度：{len(self.data_list)}')
        self.task_num = 12

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def get_dataset_name(self):
        return 'QM9'

    @property
    def processed_file_names(self):
        return 'qm9_pyg.pt'

    def get(self, idx: int) -> Data:
        cur_data = self.data_list[idx]
        # smiles = cur_data.smiles
        # mol = Chem.Mol(Chem.MolFromSmiles(str(smiles)))
        # fp = getFingerprintsFeature(mol, smiles)
        # cur_data['fp'] = fp
        # if config.use_rdkit_feature:
        #     rdkit_feature = rdkit_2d_features_generator(mol)
        #     cur_data['rdkit_feature'] = rdkit_feature
        return cur_data

    def len(self) -> int:
        return self.data_length

    def process(self):
        if not os.path.exists(os.path.join(self.processed_dir, 'id_to_smiles.pkl')):
            data2 = pd.read_csv(os.path.join(self.root, 'qm9.csv'))
            id_to_smiles = {}
            for item in data2.iterrows():
                item = item[1]
                name = item[0]
                smiles = item[1]
                id_to_smiles[name] = smiles
                # print(name, smiles)
            with open(os.path.join(self.processed_dir, 'id_to_smiles.pkl'), 'wb') as f:
                pickle.dump(id_to_smiles, f)

        with open(os.path.join(self.processed_dir, 'id_to_smiles.pkl'), 'rb') as f:
            id_to_smiles = pickle.load(f)

        R = data_ev['R']
        Z = data_ev['Z']
        N = data_ev['N']
        mole_id = data_ev['id'] + 1
        split = np.cumsum(N)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z, split)
        target = {}
        for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']:
            target[name] = np.expand_dims(data_ev[name], axis=-1)
        # y = np.expand_dims([data[name] for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']], axis=-1)

        data_list = []
        for i in tqdm(range(len(N)), desc='正在处理QM9'):
            mol_name = f'gdb_{mole_id[i]}'
            R_i = torch.tensor(R_qm9[i], dtype=torch.float32)
            z_i = torch.tensor(Z_qm9[i], dtype=torch.int64)
            y_i = [target[name][i].item() for name in
                   ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']]
            cur_y = torch.tensor(np.array(y_i))

            smiles = id_to_smiles[mol_name]
            # print(mol_name, smiles)
            try:
                mol = pyg.utils.from_smiles(smiles)
            except Exception as e:
                print('ERROR：无法解析的 smiles 串')
                continue

            edge_index = torch.tensor(mol.edge_index, dtype=torch.long)

            mol = Chem.Mol(Chem.MolFromSmiles(str(smiles)))
            mol = Chem.Mol(Chem.AddHs(mol))

            # 获取节点特征
            cur_node_feature = AtomFeature().get_node_feature(mol, smiles)

            # 获取边特征
            cur_edge_feature = BondFeature().get_bond_feature(mol, edge_index)

            cur_node_feature = torch.tensor(cur_node_feature, dtype=torch.float32)
            cur_edge_feature = torch.tensor(cur_edge_feature, dtype=torch.float32)

            idx = smi2index.get(smiles)
            if idx is None:
                print('Error！没有获取到改原子的特征文件！')
                continue
            cur_fp = smiles_fp[idx]

            cur_graph = Data(x=cur_node_feature,
                             edge_attr=cur_edge_feature,
                             y=cur_y,
                             pos=R_i,
                             name=name,
                             idx=i,
                             z=z_i,
                             edge_index=edge_index,
                             smiles=smiles,
                             fp=cur_fp)

            data_list.append(cur_graph)

        torch.save(data_list, self.processed_paths[0])
