import math

import torch
from torch import Tensor
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_undirected

from .concepts import Conceptizator

class EntropyLinear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, n_classes: int, temperature: float = 0.6,
                 bias: bool = True, conceptizator: str = 'identity_bool') -> None:
        super(EntropyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.conceptizator = Conceptizator(conceptizator)
        self.alpha = None
        self.weight = nn.Parameter(torch.Tensor(n_classes, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_classes, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        self.conceptizator.concepts = input
        # compute concept-awareness scores
        gamma = self.weight.norm(dim=1, p=1)
        self.alpha = torch.exp(gamma/self.temperature) / torch.sum(torch.exp(gamma/self.temperature), dim=1, keepdim=True)

        # weight the input concepts by awareness scores
        self.alpha_norm = self.alpha / self.alpha.max(dim=1)[0].unsqueeze(1)
        self.concept_mask = self.alpha_norm > 0.5
        x = input.multiply(self.alpha_norm.unsqueeze(1))

        # compute linear map
        x = x.matmul(self.weight.permute(0, 2, 1)) + self.bias
        return x.permute(1, 0, 2)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_classes={}'.format(
            self.in_features, self.out_features, self.n_classes
        )


def dense_to_sparse_with_attr(x, adj):
    x = x.reshape(-1, x.shape[-1])
    # adj2 = (adj.abs()+0.001).sum(dim=-1)
    # index = adj2.nonzero(as_tuple=True)
    # edge_attr = adj[index]
    # batch = index[0] * adj.size(-2)
    # index = (batch + index[0], batch + index[1])
    # edge_index = torch.stack(index, dim=0)
    # edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    # batch = torch.div(batch, adj.size(-2), rounding_mode='floor')
    index = torch.LongTensor([[i, j] for i in range(len(adj)) for j in range(len(adj))]).T
    batch = index[0]
    batch_mul = index[0] * adj.size(-2)
    index = (batch_mul + index[0], batch_mul + index[1])
    edge_index = torch.stack(index, dim=0)
    edge_index = to_undirected(edge_index, None)
    return x, edge_index, None, batch


class LogDiffPool(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, compute_link_loss: bool = True) -> None:
        super(LogDiffPool, self).__init__()
        self.alpha_r = None
        self.alpha_c = None
        self.gamma = None
        self.compute_link_loss = compute_link_loss
        self.link_loss = 0
        self.entropy_loss = 0
        self.adj = None

    def forward(self, embed: Tensor, pool: Tensor, edge_index: Tensor, batch_index: Tensor = None, adj: Tensor = None) -> (Tensor, Tensor):
        assert not (batch_index is None and adj is None)

        if batch_index is not None:
            adj = to_dense_adj(edge_index, batch_index)
            embed, embfake = to_dense_batch(embed, batch_index)
            pool, poolfake = to_dense_batch(pool, batch_index)

        embed = embed.unsqueeze(0) if embed.dim() == 2 else embed
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        pool = pool.unsqueeze(0) if pool.dim() == 2 else pool
        self.adj = adj

        self.alpha_r = torch.softmax(pool, dim=-1)
        self.alpha_c = torch.softmax(pool, dim=-2)
        self.gamma = self.alpha_r * self.alpha_c

        out = torch.matmul(self.gamma.transpose(1, 2), embed)
        out_adj = torch.matmul(torch.matmul(self.gamma.transpose(1, 2), adj), self.gamma)

        out, out_adj, edge_attr, batch = dense_to_sparse_with_attr(out, out_adj)
        return out, edge_index, edge_attr, batch_index


if __name__ == '__main__':
    embed = torch.randn((10, 2))
    adj = torch.randint(0, 2, (10, 10)).float()
    pool = torch.randint(-5, 5, (10, 3)).float()

    ldp = LogDiffPool()
    out, out_adj = ldp(embed, adj, pool)

    data = torch.rand((10, 5))
    layer = EntropyLinear(5, 4, 2)
    out = layer(data)
    print(out.shape)

