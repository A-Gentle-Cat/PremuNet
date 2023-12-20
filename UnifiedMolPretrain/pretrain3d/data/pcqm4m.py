import os
import os.path as osp
import pickle
import shutil
from pretrain3d.utils.graph import smiles2graphwithface
from pretrain3d.utils.torch_util import replace_numpy_with_torchtensor
from pretrain3d.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import re
from torch_sparse import SparseTensor
from rdkit import Chem
from copy import deepcopy
from torch_geometric.data import Data
from rdkit.Chem.rdmolfiles import SDMolSupplier
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split


from pretrain3d.utils.gt import isomorphic_core


class PCQM4Mv2Dataset(InMemoryDataset):
    def __init__(
            self,
            root="./dataset",
            smiles2graph=smiles2graphwithface,
            transform=None,
            pre_transform=None,
            xyzdir="./data/data/pcqm4m-v2_xyz",
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "pcqm4m-v2")
        self.version = 1

        self.url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"

        if osp.isdir(self.folder) and (
                not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print("PCQM4Mv2 dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        self.xyzdir = xyzdir

        super().__init__(self.folder, transform, pre_transform)
        # self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        sdf_file = "./dataset/pcqm4m-v2-train.sdf"
        sup = SDMolSupplier(sdf_file)

        print("读取sdf完成")
        print('开始生成smiles串')
        mol_list = []
        smiles_list = []
        for i, mol in tqdm(enumerate(sup)):
            try:
                self.smiles2graph(mol)
            except Exception as e:
                print(e)
                continue
            smiles_list.append(Chem.MolToSmiles(mol))
            mol_list.append(mol)
        
        print("生成smiles串完成")
        with open('./dataset/smiles_list.pkl', 'wb') as f:
            pickle.dump(smiles_list, f)
        with open('./dataset/mol_list.pkl', 'wb') as f:
            pickle.dump(mol_list, f)

        with open('./dataset/smiles_list.pkl', 'rb') as f:
            smiles_list = pickle.load(f)
        with open('./dataset/mol_list.pkl', 'rb') as f:
            mol_list = pickle.load(f)


        # data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        # smiles_list = data_df["smiles"]
        # homolumogap_list = data_df["homolumogap"]

        # split_dict = {}
        # split_dict["train"], split_dict[''] = train_test_split(range(len(smiles_list)), test_size=0.05)

        split_dict = self.get_idx_split(len(smiles_list))
        train_idxs = split_dict["train"].tolist()
        train_idxs = set(train_idxs)
        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = DGData()

            smiles = smiles_list[i]

            if i in train_idxs:
                mol = mol_list[i]
                pos = mol.GetConformer(0).GetPositions()
            else:
                mol = Chem.MolFromSmiles(smiles)
                num_atoms = mol.GetNumAtoms()
                pos = np.zeros((num_atoms, 3), dtype=np.float)

            graph = self.smiles2graph(mol)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            # data.y = torch.Tensor([homolumogap])
            data.y = torch.Tensor([0])

            data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
            data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
            data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
            data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
            data.num_rings = int(graph["num_rings"])
            data.n_edges = int(graph["n_edges"])
            data.n_nodes = int(graph["n_nodes"])
            data.n_nfs = int(graph["n_nfs"])

            data.pos = torch.from_numpy(pos).to(torch.float)
            data.rdmol = deepcopy(mol)

            # mol.RemoveAllConformers()
            # Chem.AddHs(mol)
            # ret_id = AllChem.EmbedMolecule(mol, maxAttempts=1000)
            # if ret_id < 0:
            #     ret_id = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=1000)
            # try:
            #     AllChem.MMFFOptimizeMolecule(mol)
            #     Chem.RemoveHs(mol)
            #     rdpos = mol.GetConformer(0).GetPositions()
            # except:
            #     print("failed to embed pos with rdkit")
            #     rdpos = np.random.randn(*pos.shape)
            # assert rdpos.shape[0] == pos.shape[0]
            # data.rdpos = torch.from_numpy(rdpos).to(torch.float)

            data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
            data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
            data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)

            data.isomorphisms = isomorphic_core(mol)

            data_list.append(data)

        assert all([not torch.isnan(data_list[i].y)[0] for i in split_dict["train"]])
        assert all([not torch.isnan(data_list[i].y)[0] for i in split_dict["valid"]])
        # assert all([torch.isnan(data_list[i].y)[0] for i in split_dict["test-dev"]])
        # assert all([torch.isnan(data_list[i].y)[0] for i in split_dict["test-challenge"]])

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, len):
        # split_dict = replace_numpy_with_torchtensor(
        #     torch.load(osp.join(self.root, "split_dict.pt"))
        # )
        split_dict = {}
        split_dict['train'], split_dict['valid'] = train_test_split(range(len), test_size=0.05)
        split_dict['train'] = np.array(split_dict['train'])
        split_dict['valid'] = np.array(split_dict['valid'])
        return split_dict


class DGData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face)", key)):
            return -1
        elif bool(re.search("(nf_node|nf_ring|nei_tgt_mask)", key)):
            return -1
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        if bool(re.search("(ring_index|nf_ring)", key)):
            return int(self.num_rings.item())
        elif bool(re.search("(index|face|nf_node)", key)):
            return self.num_nodes
        else:
            return 0
