from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import unbatch
from torch_scatter import scatter


class PreprocessBatch:
    def __init__(self, norm2origin=True) -> None:
        self.norm2origin = norm2origin

    def process(self, pos, batch, n_nodes):
        if not self.norm2origin:
            return
        if self.norm2origin:
            pos_min = self.global_min_pool(pos, batch)
            pos_max = global_max_pool(pos, batch)
            pos_mean = global_mean_pool(pos, batch)

            pos_min = torch.repeat_interleave(pos_min, n_nodes, dim=0)
            pos_max = torch.repeat_interleave(pos_max, n_nodes, dim=0)
            pos_mean = torch.repeat_interleave(pos_mean, n_nodes, dim=0)
            # print(f'eq')
            # if torch.any(torch.eq(pos_max-pos_min, torch.zeros(pos_max.shape, device=config.device))):
            #     print('=============================================================')
            #     print(pos_max[0])
            #     print(pos_min[0])
            #     print(pos)
            #     print(pos_max-pos_min)
            #     print('=============================================================')
            pos = (pos - pos_min) / (pos_max - pos_min + 0.0001) * 2.0 + (-1.0)
        return pos

    def global_min_pool(self, x: Tensor, batch: Optional[Tensor],
                        size: Optional[int] = None) -> Tensor:
        if batch is None:
            return x.min(dim=-2, keepdim=x.dim() == 2)
        size = int(batch.max().item() + 1) if size is None else size
        return scatter(x, batch, dim=-2, dim_size=size, reduce='min')


if __name__ == '__main__':
    processor = PreprocessBatch()
    pos = torch.tensor([[4, 4], [6, 6], [-5, -5], [3, 3], [15, 15], [-13, -13], [12, 12]], dtype=torch.float32)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    n_nodes = torch.tensor([4, 3])
    pos = processor.process(pos, batch, n_nodes)
    print(pos)
