import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Linear, ModuleList
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import OptTensor

from Models.Layers.Readout import GlobalAttention
from Models.Layers.pnaConv import PNAConv
import config


class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        pnanet=False,
        **kwargs,
    ):
        super().__init__()

        self.pnanet = pnanet
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        self.atom_agg = nn.Linear(config.pretrained_atom_feature_size, config.embedded_pretrained_feature_size)

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()

        if config.use_pretrained_atom_feature and not self.pnanet:
            # in_channels += config.pretrained_atom_feature_size
            in_channels += config.embedded_pretrained_feature_size

        if num_layers > 1:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))

        self.norms = None
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                hidden_channels,
                **(norm_kwargs or {}),
            )
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm_layer))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

        if config.aggr_type == 'attention':
            self.readout = GlobalAttention(in_feature=out_channels,
                                           hidden_size=config.mid_size,
                                           out_feature=config.attn_outsize)
            self.att_lin1 = nn.Linear(in_features=config.attn_outsize*out_channels,
                                      out_features=config.mid_size)
            self.att_lin2 = nn.Linear(in_features=config.mid_size,
                                      out_features=out_channels)
        elif config.aggr_type == 'sum':
            self.readout = global_add_pool
        elif config.aggr_type == 'mean':
            self.readout = global_mean_pool
        elif config.aggr_type == 'max':
            self.readout = global_max_pool
        else:
            self.readout = global_add_pool

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(
        self, X,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        X = X.to(config.device)
        x, edge_index, edge_attr = X.x, X.edge_index, X.edge_attr
        if config.use_pretrained_atom_feature and not self.pnanet:
            pre_x = self.atom_agg(X.pre_x)
            x = torch.concat([X.x, pre_x], dim=1)
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight,
                                  edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x

        x = self.readout(x, batch=X.batch)
        if config.aggr_type == 'attention':
            x = self.att_lin2(torch.relu(self.att_lin1(x)))

        return x

    @torch.no_grad()
    def inference(self, loader: NeighborLoader,
                  device: Optional[torch.device] = None,
                  progress_bar: bool = False) -> Tensor:
        r"""Performs layer-wise inference on large-graphs using
        :class:`~torch_geometric.loader.NeighborLoader`.
        :class:`~torch_geometric.loader.NeighborLoader` should sample the the
        full neighborhood for only one layer.
        This is an efficient way to compute the output embeddings for all
        nodes in the graph.
        Only applicable in case :obj:`jk=None` or `jk='last'`.
        """
        assert self.jk_mode is None or self.jk_mode == 'last'
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert len(loader.node_sampler.num_neighbors) == 1
        assert not self.training
        # assert not loader.shuffle  # TODO (matthias) does not work :(
        if progress_bar:
            pbar = tqdm(total=len(self.convs) * len(loader))
            pbar.set_description('Inference')

        x_all = loader.data.x.cpu()
        loader.data.n_id = torch.arange(x_all.size(0))

        for i in range(self.num_layers):
            xs: List[Tensor] = []
            for batch in loader:
                x = x_all[batch.n_id].to(device)
                if hasattr(batch, 'adj_t'):
                    edge_index = batch.adj_t.to(device)
                else:
                    edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index)[:batch.batch_size]
                if i == self.num_layers - 1 and self.jk_mode is None:
                    xs.append(x.cpu())
                    if progress_bar:
                        pbar.update(1)
                    continue
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.norms is not None:
                    x = self.norms[i](x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                if i == self.num_layers - 1 and hasattr(self, 'lin'):
                    x = self.lin(x)
                xs.append(x.cpu())
                if progress_bar:
                    pbar.update(1)
            x_all = torch.cat(xs, dim=0)
        if progress_bar:
            pbar.close()
        del loader.data.n_id

        return x_all

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')
class PNA(BasicGNN):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.PNAConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return PNAConv(in_channels, out_channels, **kwargs)