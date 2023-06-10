import torch
import torch.nn as nn
from rdkit import RDLogger
import config

RDLogger.DisableLog('rdApp.*')


class FPN(nn.Module):
    def __init__(self, in_channals, hidden_channals, mid_channals, out_channals, drop_p):
        super(FPN, self).__init__()
        self.in_channals = in_channals
        self.lin1 = nn.Linear(in_channals, hidden_channals)
        self.lin2 = nn.Linear(hidden_channals, mid_channals)
        self.lin3 = nn.Linear(mid_channals, out_channals)
        # self.drop = nn.Dropout(p=drop_p)
        self.relu = nn.ReLU()

    def forward(self, X):
        # (batchsize, fingerprints size)
        if config.concat_two_fingerprints:
            fp = torch.concat([X.tsfm_fp.view(-1, config.fingerprints_size_trans), X.traditional_fp.view(-1, config.fingerprints_size_ecfp)], dim=1).to(config.device)
        elif config.fingerprints_catogory == 'trans':
            fp = X.tsfm_fp.to(config.device)
        else:
            fp = X.traditional_fp.to(config.device)
        # print(f'tsfm: {X.tsfm_fp.view(-1, config.fingerprints_size_trans).shape} tradi: {X.traditional_fp.view(-1, config.fingerprints_size_ecfp).shape}')
        # print(f'fp.shape: {fp.shape}')
        fp = fp.view(-1, self.in_channals)
        # print('================================================')
        # print(torch.isnan(fp).any())
        # print(torch.isnan(self.lin1.weight.data).any())
        # print(torch.max(self.lin1.weight.data))
        # print(torch.min(self.lin1.weight.data))
        # print(torch.mean(self.lin1.weight.data))
        hidden_feature = self.lin1(fp)
        hidden_feature = self.relu(hidden_feature)
        out = torch.relu(self.lin2(hidden_feature))
        out = self.lin3(out)

        return out

class SMILES_Transformer(nn.Module):
    def __init__(self, in_channals, hidden_channals, mid_channals, out_channals, drop_p):
        super(SMILES_Transformer, self).__init__()
        self.in_channals = in_channals
        self.lin1 = nn.Linear(in_channals, hidden_channals)
        self.lin2 = nn.Linear(hidden_channals, mid_channals)
        self.lin3 = nn.Linear(mid_channals, out_channals)
        # self.drop = nn.Dropout(p=drop_p)
        self.relu = nn.ReLU()

    def forward(self, X):
        fp = X.tsfm_fp.to(config.device)
        fp = fp.view(-1, self.in_channals)
        hidden_feature = self.lin1(fp)
        hidden_feature = self.relu(hidden_feature)
        out = torch.relu(self.lin2(hidden_feature))
        out = self.lin3(out)

        return out
