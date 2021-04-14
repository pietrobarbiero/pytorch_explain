import math

import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter, init
from torch.nn.utils import prune
import torch.nn.functional as F

from .concepts import Conceptizator


class Logic(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, activation: str,
                 bias: bool = True, top: bool = False) -> None:
        super(Logic, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.top = top
        self.conceptizator = Conceptizator(activation)
        self.activation = activation

    def forward(self, input: Tensor) -> Tensor:
        x = self.conceptizator(input)
        if not self.top:
            x = torch.nn.functional.linear(x, self.weight, self.bias)
        return x

    def extra_repr(self) -> str:
        return 'conceptizator={}, in_features={}, out_features={}, bias={}'.format(
            self.conceptizator, self.in_features, self.out_features, self.bias is not None
        )


class LogicAttention(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, n_classes: int, n_heads: int = None,
                 top: bool = False, bias: bool = True) -> None:
        super(LogicAttention, self).__init__(in_features, out_features, bias)
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.top = top
        self.conceptizator = Conceptizator('identity_bool')
        if n_heads is not None:
            self.shrink = True
            self.weight = Parameter(torch.Tensor(n_classes, n_heads, out_features, in_features))
            if bias:
                self.bias = Parameter(torch.Tensor(n_classes, n_heads, 1, out_features))
            else:
                self.register_parameter('bias', None)
        else:
            self.shrink = False
            self.weight = Parameter(torch.Tensor(n_classes, out_features, in_features))
            if bias:
                self.bias = Parameter(torch.Tensor(n_classes, 1, out_features))
            else:
                self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        self.conceptizator.concepts = input
        if self.shrink:
            input = input.unsqueeze(0)
            x = input.matmul(self.weight.permute(0, 1, 3, 2)) + self.bias
            x = x.mean(dim=1)
        else:
            x = input.matmul(self.weight.permute(0, 2, 1)) + self.bias
        if self.top:
            x = x.view(self.n_classes, -1).t()
            self.conceptizator.concepts = x
        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_classes={}, shrink={}, top={}, conceptizator={}'.format(
            self.in_features, self.out_features, self.n_classes, self.shrink, self.top, self.activation
        )


class DisentangledConcepts(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features_per_concept: int, n_concepts: int, out_features_per_concept: int, bias: bool = True) -> None:
        super(DisentangledConcepts, self).__init__(n_concepts * in_features_per_concept,
                                                   n_concepts * out_features_per_concept,
                                                   bias)
        self.in_features_per_concept = in_features_per_concept
        self.n_concepts = n_concepts
        self.out_features_per_concept = out_features_per_concept
        self._prune()

    def _prune(self):
        blocks = []
        block_size = (self.out_features_per_concept, self.in_features_per_concept)
        for i in range(self.n_concepts):
            blocks.append(torch.ones(block_size))

        mask = torch.block_diag(*blocks)
        prune.custom_from_mask(self, name="weight", mask=mask)
        return

    def extra_repr(self) -> str:
        return 'in_features={}, n_concepts={}, out_features={}, bias={}'.format(
            self.in_features, self.n_concepts, self.out_features, self.bias is not None
        )


if __name__ == '__main__':
    data = torch.rand((10, 5))
    layer = LogicAttention(5, 4, 2)
    out = layer(data)
    print(out.shape)
    layer2 = LogicAttention(4, 3, 2)
    out2 = layer2(out)
    print(out2.shape)
    layer2 = LogicAttention(3, 1, 2)
    out3 = layer2(out2).view(-1, 2)
    print(out3.shape)
    print(out3)
