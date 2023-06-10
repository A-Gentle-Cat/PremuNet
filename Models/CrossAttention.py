import math

import torch
from torch import nn as nn

class CrossAttention(nn.Module):
    def __init__(self, input_size, attn_hidden):
        super().__init__()
        self.W_k = nn.Linear(input_size, attn_hidden)
        self.W_q = nn.Linear(input_size, attn_hidden)
        self.W_v = nn.Linear(input_size, attn_hidden)
        self.dim = attn_hidden

    def forward(self, query_x, value_x):
        """
        @param query_x: (num_atom, hidden_dim)
        @param value_x: (num_atom, hidden_dim)
        @return: (num_atom, hidden_dim)
        """
        query = self.W_q(query_x)
        key = self.W_k(value_x)
        value = self.W_v(value_x)

        key = torch.transpose(key, 0, 1)
        # print(f'query: {query.shape} key: {key.shape} value: {value.shape}')
        score_mtr = torch.div(torch.matmul(query, key), math.sqrt(self.dim))
        score_mtr = torch.softmax(score_mtr, dim=-1)

        res = torch.matmul(score_mtr, value)

        return res
