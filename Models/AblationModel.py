import torch
from torch import nn

import config


class FPN_Tradi(nn.Module):
    def __init__(self, in_channals, hidden_channals, mid_channals, out_channals, drop_p):
        super(FPN_Tradi, self).__init__()
        self.in_channals = in_channals
        self.lin1 = nn.Linear(in_channals, hidden_channals)
        self.lin2 = nn.Linear(hidden_channals, mid_channals)
        self.lin3 = nn.Linear(mid_channals, out_channals)
        # self.drop = nn.Dropout(p=drop_p)
        self.relu = nn.ReLU()

    def forward(self, X):
        fp = X.traditional_fp.to(config.device)
        fp = fp.view(-1, self.in_channals)
        hidden_feature = self.lin1(fp)
        hidden_feature = self.relu(hidden_feature)
        out = torch.relu(self.lin2(hidden_feature))
        out = self.lin3(out)

        return out