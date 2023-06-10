import torch_geometric as pyg

from FeatureDefine.AtomFeature import AtomFeature
from FeatureDefine.BondFeature import BondFeature
from torch_geometric import data as pdata

from config import *
from FeatureDefine.RDKitFeature import rdkit_2d_features_generator
from Utils.load_utils import *
from FeatureDefine.Conformer3DFeature import *

os.chdir(root_dir)

class QM8_molDataset(pdata.InMemoryDataset):
    def __init__(self):
        self.dataset_name = 'QM8'
        self.data_paths = f'./dataset/{self.dataset_name}/{self.dataset_name}.csv'
        self.data_dir = f'./dataset/{self.dataset_name}/'
        self.task_num = 16

        print(f'Processed Paths: {self.processed_paths[0]}')
        if config.reset:
            self.process()
        super(QM8_molDataset, self).__init__()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return f"./dataset/QM8/qm8_sdf"

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_dir(self) -> str:
        return f'./dataset/{self.dataset_name}/processed'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'

    @property
    def get_dataset_name(self):
        return f'{self.dataset_name} DataSet'

    def download(self):
        pass

    def process(self):
        graph_list = []

        mols = Chem.SDMolSupplier('./whl/qm8.sdf')
        csv_data = pd.read_csv('./whl/qm8.sdf.csv')
        for i in range(len(mols)):
            try:
                mol = mols[i]
                smiles = Chem.MolToSmiles(mol)
                pos = mol.GetConformers()[0].GetPositions()
                print(pos.shape)
                print(mol.GetNumAtoms())
                print(mol.GetPropsAsDict())
                print(list(mol.GetPropNames()))
            except Exception as e:
                print(e)
                print('解析错误')
                continue
            cur_pos = np.array(pos)
            item = csv_data.iloc[i]
            label = item[1:]

            if str(smiles).find('*') != -1:
                print('Error: smiles 中发现了*')
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print('rdkit ERROR：无法解析的 smiles 串', smiles)
                continue

            try:
                mol = pyg.utils.from_smiles(smiles)
            except Exception as e:
                print('pyg ERROR：无法解析的 smiles 串', smiles)
                continue

            edge_index = torch.tensor(mol.edge_index, dtype=torch.long)

            mol = Chem.Mol(Chem.MolFromSmiles(str(smiles)))
            mol = Chem.Mol(Chem.AddHs(mol))

            # 获取节点特征
            cur_node_feature = AtomFeature().get_node_feature(mol, smiles)
            z = AtomFeature().get_atom_type(mol)

            # 获取边特征
            cur_edge_feature = BondFeature().get_bond_feature(mol, edge_index)

            cur_node_feature = torch.tensor(cur_node_feature, dtype=torch.float32)
            cur_edge_feature = torch.tensor(cur_edge_feature, dtype=torch.float32)

            cur_label = np.array(label)

            cur_fp = smiles_fp[idx]

            cur_graph = pdata.Data(x=cur_node_feature,
                                   z=z,
                                   edge_attr=cur_edge_feature,
                                   y=cur_label,
                                   edge_index=edge_index,
                                   smiles=smiles,
                                   pos=cur_pos,
                                   fp=cur_fp)
            if config.use_rdkit_feature:
                rdkit_feature = rdkit_2d_features_generator(mol)
                cur_graph['rdkit_feature'] = rdkit_feature

            graph_list.append(cur_graph)

        random.seed(config.seed)
        random.shuffle(graph_list)

        torch.save(self.collate(graph_list), self.processed_paths[0])


