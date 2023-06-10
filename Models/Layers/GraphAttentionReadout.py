from Models.GCN_Net import *

class GraphAttentionReadout(nn.Module):
    def __init__(self, in_dim, att_dim, r_dim=1):
        super().__init__()
        self.W1 = nn.Linear(in_dim, att_dim)
        self.W2 = nn.Linear(att_dim, r_dim)

    def attention_read_out(self, x, batch):
        cur = 0
        H = []
        h = []
        for i, item in enumerate(x):
            if batch[i] > cur:
                cur = batch[i]
                h = torch.stack(h, dim=0)
                H.append(h)
                h = []
            h.append(item)
            if i == x.shape[0] - 1:
                H.append(h)

        return H

    def forward(self, x, batch):
        H = self.attention_read_out(x, batch)
        res = torch.zeros((len(H), H[0].shape[0]*H[0].shape[1]))
        print('=========')
        print(H[0].shape)
        for i, h in enumerate(H):
            S = torch.softmax(self.W2(torch.tanh(self.W1(h))), dim=1)
            h = torch.tensor(h)
            print(h.shape)
            out = torch.flatten(S * h)
            res[i] = out

        return res

