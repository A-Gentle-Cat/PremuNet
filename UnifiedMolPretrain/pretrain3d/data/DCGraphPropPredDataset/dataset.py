import pandas as pd
from torch_geometric.data import InMemoryDataset
import shutil, os
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
from UnifiedMolPretrain.pretrain3d.data.pcqm4m import DGData
from UnifiedMolPretrain.pretrain3d.model.gnn import one_hot_atoms, one_hot_bonds
from UnifiedMolPretrain.pretrain3d.utils.graph import smiles2graphwithface
from rdkit import Chem
from copy import deepcopy
from UnifiedMolPretrain.pretrain3d.data.DCGraphPropPredDataset.deepchem_dataloader import (
    load_molnet_dataset,
    get_task_type,
)


class DCGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root="./unified_dataset", transform=None, pre_transform=None):
        assert name.startswith("dc-")
        name = name[len("dc-"):]
        self.name = name
        self.dirname = f"dcgraphproppred_{name}"
        self.original_root = root
        self.root = osp.join(root, self.dirname)
        super().__init__(self.root, transform, pre_transform)
        self.process()
        self.data, self.slices, self._num_tasks = torch.load(self.processed_paths[0])

    def get_idx_split(self):
        path = os.path.join(self.root, "split", "split_dict.pt")
        return torch.load(path)

    @property
    def task_type(self):
        return get_task_type(self.name)

    @property
    def eval_metric(self):
        return "rocauc" if "classification" in self.task_type else "mae"

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def raw_file_names(self):
        return ["data.npz"]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    @property
    def raw_dir(self) -> str:
        return f"./dataset"

    def download(self):
        pass

    def process(self):
        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []
        _, dfs, _ = load_molnet_dataset(self.name)
        data = pd.read_csv('./dataset/BBBP/mapping/mol.csv')
        smiles_list = data['smiles']
        labels_list = data['p_np']

        num_tasks = len(dfs[0]["labels"].values[0])

        for smiles, labels in tqdm(zip(smiles_list, labels_list)):
            data = DGData()
            mol = Chem.MolFromSmiles(smiles)
            graph = smiles2graphwithface(mol)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = one_hot_bonds(torch.from_numpy(graph["edge_feat"]).to(torch.int64))
            data.x = one_hot_atoms(torch.from_numpy(graph["node_feat"]).to(torch.int64))

            if "classification" in self.task_type:
                data.y = torch.as_tensor(labels).view(1, -1).to(torch.long)
            else:
                data.y = torch.as_tensor(labels).view(1, -1).to(torch.float32)

            # data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
            # data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
            # data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
            # data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
            # data.num_rings = int(graph["num_rings"])
            data.n_edges = int(graph["n_edges"])
            data.n_nodes = int(graph["n_nodes"])
            # data.n_nfs = int(graph["n_nfs"])
            # data.rdmol = deepcopy(mol)

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices, num_tasks), self.processed_paths[0])



if __name__ == "__main__":
    dataset = DCGraphPropPredDataset("dc-bbbp")
    split_index = dataset.get_idx_split()
    print(split_index)
