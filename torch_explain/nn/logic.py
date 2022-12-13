import math

import torch
from torch import Tensor
from torch import nn

from ..logic.semantics import ProductTNorm, Logic
from ..logic.parser import ExpressionTree, Concept, Not, And, Or, serialize_rules
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
        self.alpha = torch.exp(gamma / self.temperature) / torch.sum(torch.exp(gamma / self.temperature), dim=1,
                                                                     keepdim=True)

        # weight the input concepts by awareness scores
        self.alpha_norm = self.alpha / self.alpha.max(dim=1)[0].unsqueeze(1)
        self.concept_mask = self.alpha_norm > 0.5
        # if len(input.shape) == 3:
        #     x = input.unsqueeze(1).multiply(self.alpha_norm.unsqueeze(0).unsqueeze(-1))
        #     x = torch.einsum('btce,thc->bteh', x, self.weight)
        #     x = x.reshape(-1, x.shape[-1])
        # else:
        x = input.multiply(self.alpha_norm.unsqueeze(1))
        x = x.matmul(self.weight.permute(0, 2, 1)) + self.bias
        x = x.permute(1, 0, 2)

        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_classes={}'.format(
            self.in_features, self.out_features, self.n_classes
        )


class PropositionalLayer(nn.Module):
    def __init__(self):
        super(PropositionalLayer, self).__init__()

    def forward(self, x, expression_tree: ExpressionTree, logic: Logic):
        # logic.update()    # TODO: check if we really need to train logic
        tasks = []
        for r in expression_tree.roots:
            tasks.append(self._visit(r, x, logic))
        return torch.concat(tasks, dim=1)

    def _visit(self, node, x, logic):
        # for each node in the expression tree either:
        # - return a concept (leaf) or
        # - perform logic composition of child nodes
        if isinstance(node, Concept):
            return x[:, node.id:node.id + 1]
        else:
            visited = []
            for c in node.children:
                x_viz = self._visit(c, x, logic)
                visited.append(x_viz)
            visited = torch.concat(visited, dim=1)
            if isinstance(node, Not):
                ops_result = logic.neg(visited)
            elif isinstance(node, And) and visited.shape[1] > 1:
                ops_result = logic.conj(visited)
            elif isinstance(node, Or) and visited.shape[1] > 1:
                ops_result = logic.disj(visited)
            else:
                raise NotImplementedError
            return ops_result


if __name__ == '__main__':
    data = torch.rand((10, 5))
    layer = EntropyLinear(5, 4, 2)
    out = layer(data)
    print(out.shape)
