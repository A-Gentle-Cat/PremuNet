import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, precision_recall_curve, auc
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm

from pretrain_trfm import TrfmSeq2seq
from utils.Smiles2token import Token2Idx, NewDic
from utils.Smiles2token import smi_tokenizer, get_array, split


def reset_seed(seed):
    # seed = np.random.randint(0, 100000)
    print(f'Seed Reset! config.seed = 【{seed}】')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate_mlp_classification(X, y, rate, n_repeats):
    auc = np.empty(n_repeats)
    for i in range(n_repeats):
        print(f"now round: {i}")
        clf = MLPClassifier(max_iter=1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - rate, stratify=y)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        auc[i] = roc_auc_score(y_test, y_score[:, 1])
        print(f"auc: {auc[i]}")
    ret = {'auc mean': np.mean(auc), 'auc std': np.mean(np.std(auc, axis=0))}
    return ret


class Classification(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, dropout=0.1):
        super(Classification, self).__init__()
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


class FPDataLoader(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = torch.tensor(self.X[item])
        y = torch.tensor(self.y[item])
        return [x, y]


def evaluate(model, test_loader, TASKS):
    model.eval()
    total_loss = 0

    predict = torch.Tensor().cuda()
    labels = torch.Tensor().cuda()

    with torch.no_grad():
        for b, data in enumerate(test_loader):
            sm, label = data
            sm = sm.cuda()  # (T,B)
            label = label.cuda().float()

            output = model(sm)  # (T,B,V)

            predict = torch.cat((predict, output))
            labels = torch.cat((labels, label))

            use_idx = label == label
            loss = F.binary_cross_entropy_with_logits(input=output[use_idx].contiguous().view(-1),
                                                      target=label[use_idx].contiguous().view(-1))
            total_loss += loss.item()

    sig = torch.nn.Sigmoid()
    predict = sig(predict)
    labels = labels.int()

    valid_auc = 0
    for i in range(TASKS):
        use_idx = labels[:, i] == labels[:, i]
        valid_auc += roc_auc_score(labels[:, i][use_idx].cpu().detach().numpy(),
                                   np.squeeze(predict[:, i][use_idx].cpu().detach().numpy()))
    valid_auc /= TASKS

    return total_loss / len(test_loader), valid_auc


def train(X, y, dataset):
    task = y.shape[1]
    EPOCH = 1000
    learn_rate = 3e-6
    batch_size = 32
    n_worker = 4

    print("任务数：", task)
    model = Classification(1024, task, 256).cuda()

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    data_dir = f"/home/{dataset}/split/scaffold"
    train_idx = pd.read_csv(os.path.join(data_dir, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
    valid_idx = pd.read_csv(os.path.join(data_dir, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
    test_idx = pd.read_csv(os.path.join(data_dir, 'test.csv.gz'), compression='gzip', header=None).values.T[0]
    X_train, X_test, X_valid = X[train_idx], X[test_idx], X[valid_idx]
    y_train, y_test, y_valid = y[train_idx], y[test_idx], y[valid_idx]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    train_data = FPDataLoader(X_train, y_train)
    test_data = FPDataLoader(X_test, y_test)
    valid_data = FPDataLoader(X_valid, y_valid)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_worker)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=n_worker)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=n_worker)

    for e in range(1, EPOCH + 1):

        predict = torch.Tensor().cuda()
        labels = torch.Tensor().cuda()

        bar = tqdm(enumerate(train_loader))
        for b, data in bar:

            sm, label = data[0], data[1]
            # print("sm shape: ", sm.shape)
            sm = sm.cuda()  # (T,B)
            label = label.cuda().float()

            optimizer.zero_grad()

            output = model(sm)  # (T,B,V)

            predict = torch.cat((predict, output))
            labels = torch.cat((labels, label))

            use_idx = label == label
            loss = F.binary_cross_entropy_with_logits(input=output[use_idx].contiguous().view(-1),
                                                      target=label[use_idx].contiguous().view(-1))
            loss.backward()

            optimizer.step()
            bar.set_description("loss: {:.6f}".format(loss.item()))

        # print(labels.shape)
        # print(predict.shape)

        train_auc = 0
        for i in range(task):
            use_idx = labels[:, i] == labels[:, i]
            train_auc += roc_auc_score(labels[:, i][use_idx].cpu().detach().numpy(),
                                       np.squeeze(predict[:, i][use_idx].cpu().detach().numpy()))
        train_auc /= task
        # print("train_auc: ", train_auc)

        loss, auc = evaluate(model, test_loader, task)
        valid_loss, valid_auc = evaluate(model, valid_loader, task)
        print('Val {:3d} | test loss {:.6f} | train_auc {:.6f} | test auc {:.6f}'.format(e, loss, train_auc, auc))
        print(f"valid loss: {valid_loss}, valid_auc: {valid_auc}")


if __name__ == '__main__':
    trfm = TrfmSeq2seq(len(NewDic), 256, len(NewDic), 4)
    trfm.load_state_dict(torch.load('/home/trfm_12_23000.pkl'))
    trfm.eval()
    print('Total parameters:', sum(p.numel() for p in trfm.parameters()))

    ds = "tox21"
    df = pd.read_csv(f'/home/{ds}/mapping/mol.csv')

    label_list = list(df.keys())[:-1]
    label_list.remove("smiles")

    # 单独encode
    # x_split = [smi_tokenizer(sm, max_len=128, padding=True) for sm in df['smiles'].values]
    # X = np.zeros((len(x_split), 1024), dtype=np.float32)
    #
    # for i, ids in tqdm(enumerate(x_split)):
    #     X[i] = trfm.encode_one(torch.Tensor(ids).to(torch.int))

    # 一起encode
    smiles = df['smiles'].values
    x_split = [split(sm.strip()) for sm in smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    #
    # np.save("sider.npy", X)
    # X = np.load("hiv.npy")
    # print(X)
    # print(X2)

    # np.save("/home/bace.npy", X)
    # X = np.load("/home/bace.npy")
    train(X, df[label_list].values, ds)
