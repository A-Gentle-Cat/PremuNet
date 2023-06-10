import torch
from torch import nn

class GraphSequenceAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_1 = nn.Linear(embed_dim, embed_dim)
        self.W_2 = nn.Linear(embed_dim, embed_dim)
        self.W_G = nn.Linear(embed_dim, embed_dim)
        self.W_S = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, S, H):
        Ds = self.relu(self.W_1(S))
        Dh = self.relu(self.W_2(H))
        F = self.sigmoid(self.W_G(Ds) + self.W_S(Dh))

        Md = torch.mul(F, S) + torch.mul(1 - F, H)
        print(f'S.shape: {S.shape} H.shape: {H.shape} Md.shape: {Md.shape}')

        return Md
