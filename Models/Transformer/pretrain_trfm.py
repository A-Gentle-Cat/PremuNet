import argparse
import math
import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils.Smiles2token import Token2Idx


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
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(d_model=hidden_size, nhead=4,
                                   num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                   dim_feedforward=hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        hidden = self.trfm(embedded, embedded)  # (T,B,H)
        out = self.out(hidden)  # (T,B,V)
        out = F.log_softmax(out, dim=2)  # (T,B,V)
        return out  # (T,B,V)

    def _encode(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
        penul = output.detach().numpy()
        output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)  # (T,B,H)
        output = output.detach().numpy()
        # mean, max, first*2
        return np.hstack([np.mean(output, axis=0), np.max(output, axis=0), output[0, :, :], penul[0, :, :]])  # (B,4H)

    def encode(self, src):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size <= 300:
            return self._encode(src)
        else:  # Batch is too large to load
            print('There are {:d} molecules. It will take a little time.'.format(batch_size))
            st, ed = 0, 300
            out = self._encode(src[:, st:ed])  # (B,4H)
            while ed < batch_size:
                st += 300
                ed += 300
                ed = min(ed, batch_size)
                print(f"now st: {st}, now ed: {ed}")
                out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
            return out

    def encode_one(self, src):
        """
        对单个分子进行encode
        :param src: smiles串
        :return: 1024维指纹
        """
        src = src.unsqueeze(1)
        return self._encode(src)

    def encode_one_all(self, src):
        """
        得到smiles串各个部分的feature
        :param src: string
        :return: len(token) * 256
        """
        src = src.unsqueeze(1)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers):
            output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)  # (T,B,H)
        output = output.detach().numpy()
        return output


def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--data', '-d', type=str, default='/home/chembl_27.csv',
                        help='train corpus (.csv)')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')

    parser.add_argument('--n_epoch', '-e', type=int, default=50, help='number of epochs')
    # parser.add_argument('--vocab', '-v', type=str, default='chembl_vocab.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--out-dir', '-o', type=str, default='../result', help='output directory')
    # parser.add_argument('--name', '-n', type=str, default='ST', help='model name')
    parser.add_argument('--n_worker', '-w', type=int, default=8, help='number of workers')
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=4, help='number of layers')
    parser.add_argument('--n_head', type=int, default=4, help='number of attention heads')
    parser.add_argument('--lr', type=float, default=1e-5, help='Adam learning rate')
    # parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    return parser.parse_args()


def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    for b, data in enumerate(test_loader):
        mask, sm = data
        mask = torch.t(mask.cuda())  # (T,B)
        sm = torch.t(sm.cuda())  # (T,B)
        with torch.no_grad():
            output = model(mask)  # (T,B,V)
        loss = F.nll_loss(output.view(-1, len(Token2Idx)),
                          sm.contiguous().view(-1),
                          ignore_index=Token2Idx['PAD'])
        total_loss += loss.item()

    # model.train()

    return total_loss


def main():
    args = parse_arguments()
    assert torch.cuda.is_available()

    print('Loading dataset...')

    model = TrfmSeq2seq(len(Token2Idx), args.hidden, len(Token2Idx), args.n_layer).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(model)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    # smiles = pd.read_csv(args.data)['smiles'].values
    # smiles = [sme.randomize_smiles(smi) for smi in smiles]
    smiles = np.load("/home/new_smiles.npy")

    dataset = SeqDataset(smiles, max_len=128, mask_prob=0.15)
    test_size = 10000
    train, test = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)
    print('Train size:', len(train))
    print('iterator: ', len(train) / args.batch_size)
    print('Test size:', len(test))
    del dataset, train, test

    for e in range(args.n_epoch):

        bar = tqdm(enumerate(train_loader))
        for b, data in bar:
            mask, sm = data
            mask = torch.t(mask.cuda())  # (T,B)
            sm = torch.t(sm.cuda())
            optimizer.zero_grad()
            output = model(mask)  # (T,B,V)
            loss = F.nll_loss(output.view(-1, len(Token2Idx)),
                              sm.contiguous().view(-1), ignore_index=Token2Idx['PAD'])
            loss.backward()
            optimizer.step()
            if loss.item() != 0:
                bar.set_description("loss: {:.15f} | log {:.4f}".format(loss.item(), math.log10(loss.item())))
            else:
                bar.set_description("loss: {:.15f} | log：0".format(loss.item()))

            if b % 5000 == 0 and b != 0:
                if loss.item() != 0:
                    print('Train {:3d}: iter {:5d} | loss {:.15f} | log {:.4f}'.format(e, b, loss.item(),
                                                                                       math.log10(loss.item())))
                else:
                    print("Train Zero! ")
            if b % 25000 == 0 and b != 0:
                loss = evaluate(model, test_loader)
                if loss != 0:
                    print('Val {:3d}: iter {:5d} | loss {:.15f} | log {:.4f}'.format(e, b, loss, math.log10(loss)))
                else:
                    print("Valid Zero!")
                # Save the model if the validation loss is the best we've seen so far.
                # if not best_loss or loss < best_loss:
                print(f"[!] saving model...（{e}轮 {b}）")
                print(f"total loss: {loss}")
                if not os.path.isdir("save"):
                    os.makedirs("save")
                torch.save(model.state_dict(), './save/trfm_new_ds_lr5_%d_%d.pkl' % (e, b))


if __name__ == "__main__":
    log_file = open("message_lr5_new_ds.log", "w")
    sys.stdout = log_file
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
