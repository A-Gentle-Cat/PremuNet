import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss


class MV_GNN_MSE(nn.Module):
    def __init__(self, loss: nn.Module, C):
        super(MV_GNN_MSE, self).__init__()
        self.loss = loss
        self.C = C
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, out1, out2, out, target):
        loss_graph = self.loss(out1, target)
        loss_line = self.loss(out2, target)
        loss_pred = loss_graph + loss_line
        loss_dis = self.mse(out1, out2)

        res = loss_pred + self.C * loss_dis

        return res


class MV_GNN_AGG_MSE(nn.Module):
    def __init__(self, loss: nn.Module, C):
        super(MV_GNN_AGG_MSE, self).__init__()
        self.loss = loss
        self.C = C
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, out1, out2, out, target):
        loss_graph = self.loss(out, target)
        loss_dis = self.mse(out1, out2)

        res = loss_graph + self.C * loss_dis

        return res


class MV_GNN_SimCLR(nn.Module):
    def __init__(self, loss: nn.Module, C):
        super(MV_GNN_SimCLR, self).__init__()
        self.loss = loss
        self.C = C
        self.sim = NTXent(C=config.Loss_C, tau=config.tau)

    def forward(self, out1, out2, target):
        loss_graph = self.loss(out1, target)
        loss_line = self.loss(out2, target)
        loss_pred = loss_graph + loss_line
        loss_dis = self.sim(out1, out2)

        res = loss_pred + self.C * loss_dis

        return res


class NTXent(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, C: float = 1, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(NTXent, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.C = C

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum('i,j->ij', z1_abs, z2_abs) + 1e-8)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

class Sp_GNN_FP_Net_Loss(nn.Module):
    def __init__(self):
        super(Sp_GNN_FP_Net_Loss, self).__init__()
        self.loss_dis = NTXent()
        self.loss_clf = nn.BCEWithLogitsLoss()
        self.loss_regr = nn.L1Loss()
        self.C = config.Loss_C

    def forward(self, pred, out2d, out3d, target):
        loss_dis = self.loss_dis(out2d, out3d)
        if config.dataset_type == 'classification':
            loss_out = self.loss_clf(pred, target)
        else:
            loss_out = self.loss_regr(pred, target)

        return loss_out + self.C * loss_dis
