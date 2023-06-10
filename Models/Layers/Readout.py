import torch
from torch import nn
from torch_geometric.utils import unbatch

class GlobalAttention(nn.Module):
    """
       Self Attention Layer
       Given $X\in \mathbb{R}^{n \times in_feature}$, the attention is calculated by: $a=Softmax(W_2tanh(W_1X))$, where
       $W_1 \in \mathbb{R}^{hidden \times in_feature}$, $W_2 \in \mathbb{R}^{out_feature \times hidden}$.
       The final output is: $out=aX$, which is unrelated with input $n$.
    """

    def __init__(self, in_feature, hidden_size, out_feature):
        """
        The init function.
        :param hidden: the hidden dimension, can be viewed as the number of experts.
        :param in_feature: the input feature dimension.
        :param out_feature: the output feature dimension.
        """
        super(GlobalAttention, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden_size, in_feature))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(out_feature, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Use xavier_normal method to initialize parameters.
        """
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, x, batch):
        """
        The forward function.
        :param X: The input feature map. $X \in \mathbb{R}^{n \times in_feature}$.
        :return: The final embeddings and attention matrix.
        """
        out = []
        # print(x.shape)
        # print(batch)
        graphs = unbatch(x, batch=batch, dim=0)
        for graph in graphs:
            # print(f'graph: {graph.shape}')
            x = torch.tanh(torch.matmul(self.w1, graph.transpose(1, 0)))
            x = torch.matmul(self.w2, x)
            attn = torch.nn.functional.softmax(x, dim=-1)
            x = torch.matmul(attn, graph)
            # print(f'cur_out: {x.shape}')
            out.append(x.flatten())
        out = torch.stack(out, dim=0)
        return out