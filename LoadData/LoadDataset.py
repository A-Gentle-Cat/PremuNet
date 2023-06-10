import os

import torch
from torch_geometric import data as pdata
from tqdm import tqdm

from FeatureDefine.Conformer3DFeature import *
from FeatureDefine.MolFeature import MolFeature
from FeatureDefine.RDKitFeature import rdkit_2d_features_generator
from Utils.load_utils import *
import config

class molDataset(pdata.InMemoryDataset):
    def __init__(self, dataset_name, catogory, args, enhance=False, lineGraph=False, reset=False):
        self.dataset_name = dataset_name
        self.data_paths = f'./dataset/{self.dataset_name}/{self.dataset_name}.csv'
        self.data_dir = f'./dataset/{self.dataset_name}/'
        self.lineGraph = lineGraph
        self.args = args
        self.enhance = enhance
        self.catogory = catogory
        self.task_num = get_task_num(self.dataset_name)

        print(f'Processed Paths: {self.processed_paths[0]}')
        if reset:
            print('dataset reset!')
            self.process()
        super(molDataset, self).__init__()
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.smiles = [self.__getitem__(i).smiles for i in range(self.__len__())]

    @property
    def raw_dir(self) -> str:
        return f"./dataset/{self.dataset_name}/raw"

    @property
    def raw_file_names(self):
        paths = [f'{self.dataset_name}.csv']
        return paths[0]

    @property
    def processed_dir(self) -> str:
        return f'./dataset/{self.dataset_name}/processed'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        paths = ['data_train', 'data_valid', 'data_test', 'data']
        for i, path in enumerate(paths):
            paths[i] = path + '.pt'
        if self.catogory == 'train':
            return [paths[0]]
        elif self.catogory == 'valid':
            return [paths[1]]
        elif self.catogory == 'test':
            return [paths[2]]
        else:
            return [paths[3]]

    @property
    def get_dataset_name(self):
        return f'{self.dataset_name} DataSet'

    def download(self):
        pass

    def get_smiles(self):
        pass

    def process(self):
        if os.path.exists(os.path.join(self.data_dir, 'mapping', 'mol.csv')):
            data = pd.read_csv(os.path.join(self.data_dir, 'mapping', 'mol.csv'))
        else:
            data = pd.read_csv(os.path.join(self.data_dir, 'mapping', 'mol.csv.gz'), compression='gzip')

        data = [item[1] for item in data.iterrows()]
        # split_idx = get_split_idx_from_files(self.data_dir)[self.catogory]
        # data = DATASET_LOADER[self.dataset_name](data_dir=self.raw_dir, save_dir=self.raw_dir, splitter=self.splitter)
        # if self.catogory == 'train':
        #     data = data[1][0]
        # elif self.catogory == 'valid':
        #     data = data[1][1]
        # elif self.catogory == 'test':
        #     data = data[1][2]
        # elif self.catogory == 'all':
        #     data = data[1][0]
        # else:
        #     raise Exception(f'no name for type: {self.catogory}')

        graph_list = []

        for i, item in tqdm(enumerate(data)):
            smiles, label = load_smiles_and_label(config.dataset_name, item)

            # 查看是否有 GEOM 的文件
            pic_path = config.feature_3d.get(smiles)
            if pic_path is not None:
                path = os.path.join(config.path_dir, pic_path)
                mols = pickle.load(open(path, 'rb'))
                mol = mols['conformers'][0]['rd_mol']
            else:
                mol = Chem.Mol(Chem.MolFromSmiles(str(smiles)))
                mol = Chem.Mol(Chem.AddHs(mol))

            # 获取节点特征
            # cur_node_feature = AtomFeature().get_node_feature(mol, smiles)
            cur_node_feature = MolFeature.get_node_feature(mol)
            z = MolFeature.get_atoms_type(mol)

            # 获取边特征
            edge_index, cur_edge_feature = MolFeature.get_bond_feature(mol)
            # cur_edge_feature = BondFeature().get_bond_feature(mol, edge_index)

            if not isinstance(cur_edge_feature, torch.Tensor):
                cur_node_feature = torch.tensor(cur_node_feature, dtype=torch.float32)
            if not isinstance(cur_edge_feature, torch.Tensor):
                cur_edge_feature = torch.tensor(cur_edge_feature, dtype=torch.float32)
            cur_node_feature = cur_node_feature.to(torch.float32)
            cur_edge_feature = cur_edge_feature.to(torch.float32)

            cur_label = torch.tensor(np.array(label))

            # 获取 3d 特征
            idx = i
            if config.pos_3d[idx] is None:
                print(f'没有获取到 3d 坐标，将使用 0 初始化', smiles)
                cur_pos = cur_node_feature.new_zeros((1, cur_node_feature.shape[0], 3)).uniform_(-1, 1)
            else:
                cur_pos = config.pos_3d[idx]

            cur_pos = cur_pos.to(torch.float32)

            if self.args.model in ['DimeNet', 'SchNet', 'ComENet', 'SphereNet']:
                cur_pos = torch.tensor(cur_pos[0])
            else:
                cur_pos = cur_pos.numpy()
            # print(f'model: {self.args.model}, pos: {cur_pos.shape}')
            # print(f'cur_pos.shape: {cur_pos.shape}')
            # if cur_pos is None or len(cur_pos) == 0:
            #     print('Error！没有获取到该原子的 3d 坐标！')
            #     raise Exception(smiles)
            # 3d 坐标
            # cur_pos = cur_node_feature.new_zeros((cur_node_feature.shape[0], 3)).uniform_(-1, 1)
            # 获取分子指纹
            cur_tsfm_fp = config.tsfm_fp[idx]
            cur_traditional_fp = config.traditional_fp[idx]
            # 获取原子指纹
            cur_pre_atom = config.tsfm_atom_fea[idx]

            cur_graph = pdata.Data(x=cur_node_feature,
                                   z=z,
                                   edge_attr=cur_edge_feature,
                                   y=cur_label,
                                   edge_index=edge_index,
                                   smiles=smiles,
                                   tsfm_fp=cur_tsfm_fp,
                                   traditional_fp=cur_traditional_fp,
                                   pos=cur_pos,
                                   pre_x=cur_pre_atom,
                                   n_nodes=mol.GetNumAtoms(),
                                   n_edges=mol.GetNumBonds()*2)
            # print(cur_graph)
            if config.use_rdkit_feature:
                rdkit_feature = rdkit_2d_features_generator(mol)
                cur_graph['rdkit_feature'] = rdkit_feature

            graph_list.append(cur_graph)

        torch.save(self.collate(graph_list), self.processed_paths[0])

