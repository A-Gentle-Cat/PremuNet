from torch import Tensor
from torch_geometric.nn import MessagePassing

class GN_block_layers(MessagePassing):
    def __init__(self):
        super().__init__()

    def message(self, x_j: Tensor, ) -> Tensor:
        pass

    def update(self, inputs: Tensor) -> Tensor:
        pass

    def

class GN_block_tsfm(MessagePassing):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        pass
