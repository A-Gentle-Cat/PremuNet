import math

import torch
from torch import nn
from torch.autograd import Variable


#
# from utils import Token2Idx
# from utils import smi_tokenizer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # (T,H)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.0):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(d_model=hidden_size, nhead=4,
                                   num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                   dim_feedforward=hidden_size, dropout=dropout)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, X):
        src = X.token.reshape(-1, 440).T
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)

        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
        penul = output
        output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
        # print("output shape: ", output.shape)

        fp = torch.cat((torch.mean(output, dim=0), torch.max(output, dim=0)[0], output[0, :, :], penul[0, :, :]), dim=1)
        return fp  # (B,V * 4)


class Classification(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, dropout=0.0):
        super().__init__()
        self.Linear1 = nn.Linear(in_size, hidden_size)
        self.Relu = nn.ReLU()
        self.Dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_size, out_size)

        torch.nn.init.xavier_uniform_(self.Linear1.weight)
        torch.nn.init.xavier_uniform_(self.Linear2.weight)

    def forward(self, src):
        hid = self.Linear1(src)
        hid = self.Relu(hid)
        hid = self.Dropout(hid)
        return self.Linear2(hid)


class Tsfm_class(nn.Module):
    def __init__(self, hidden_size, n_layers, dic_len, task, l_hid, l_drop=0.0, tsfm_weight=None, lock=False):
        super(Tsfm_class, self).__init__()
        self.trfm = TrfmSeq2seq(dic_len, hidden_size, dic_len, n_layers)
        if tsfm_weight is not None:
            self.trfm.load_state_dict(torch.load(tsfm_weight))

        self.linear = Classification(1024, task, l_hid, l_drop)

    def forward(self, src):
        # X = X.to(config.device)
        # print(f'X.token.shape = {X.token.reshape(config.BATCH_SIZE, -1).shape}')
        # src = X.token.reshape(-1, 300).transpose(0, 1)
        # src = X.token.reshape(-1, 300)
        # src = torch.t(src)
        src = self.trfm(src)
        src = self.linear(src)
        return src