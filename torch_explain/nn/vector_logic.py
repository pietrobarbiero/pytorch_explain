import math
import torch
from torch.nn import Linear, Parameter, Module, init
from torch import Tensor
from torch.nn import functional as F


class ConceptEmbeddings(Linear):
    def __init__(self, in_features: int, out_features: int, emb_size: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConceptEmbeddings, self).__init__(in_features, out_features, bias, device, dtype)
        self.weight = Parameter(torch.empty((out_features, in_features, emb_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, emb_size, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        return (input @ self.weight).permute(1, 0, 2) + self.bias


class ConceptGate(Module):
    def __init__(self, n_concepts: int, emb_size: int, n_classes: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConceptGate, self).__init__()
        self.n_concepts = n_concepts
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.weight = Parameter(torch.empty((n_concepts, emb_size, n_classes), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(n_concepts, n_classes, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        h = (input.permute(1, 0, 2) @ self.weight).permute(1, 0, 2) + self.bias
        h = h.norm(dim=0, p=2)
        return F.softmax(h, dim=0)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return 'n_concepts={}, emb_size={}, n_classes={}, bias={}'.format(
            self.n_concepts, self.emb_size, self.n_classes, self.bias is not None
        )


def context(embedding: Tensor) -> Tensor:
    return embedding / torch.norm(embedding, p=2, dim=-1).unsqueeze(-1)


def logprobs(embedding: Tensor) -> Tensor:
    return torch.norm(embedding, p=2, dim=-1)


def semantics(embedding: Tensor) -> Tensor:
    return torch.exp(-torch.norm(embedding, p=2, dim=-1))


def to_boolean(embedding: Tensor, true_norm: float = 0, false_norm: float = 1) -> Tensor:
    sm = torch.round(semantics(embedding))
    sm[sm != 0] = false_norm
    sm[sm == 0] = true_norm
    ct = context(embedding)
    return ct * sm.unsqueeze(-1)


if __name__ == '__main__':
    n_samples = 100
    n_concepts = 5
    emb_size = 3
    n_classes = 2
    x = torch.randn(n_samples, n_concepts, emb_size)
    gate = ConceptGate(n_concepts, emb_size, n_classes)
    h = gate.forward(x)
    out = x.unsqueeze(3) * h.unsqueeze(0).unsqueeze(2)
    print(h)
    print(out.shape)
    print(h.sum(0))
