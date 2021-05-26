import math

import torch
from torch import Tensor
from torch import nn

from .concepts import Conceptizator


class ConceptAware(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, n_classes: int, temperature: float = 0.6,
                 awareness: str = 'l1', bias: bool = True, conceptizator: str = 'identity_bool') -> None:
        super(ConceptAware, self).__init__()
        self.n_classes = n_classes
        self.awareness = awareness
        self.temperature = temperature
        self.conceptizator = Conceptizator(conceptizator)
        self.alpha = None
        self.gamma = None
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
        if self.awareness == 'entropy':
            # self.alpha = torch.softmax(self.weight.norm(dim=1, p=1), dim=1)
            gamma = self.weight.norm(dim=1, p=1)
            self.alpha = torch.exp(gamma/self.temperature) / torch.sum(torch.exp(gamma/self.temperature), dim=1, keepdim=True)
        elif self.awareness == 'l1':
            self.alpha = torch.sigmoid(torch.log(self.weight.norm(dim=1, p=1)))

        # weight the input concepts by awareness scores
        alpha_norm = self.alpha / self.alpha.max(dim=1)[0].unsqueeze(1)
        self.concept_mask = alpha_norm > 0.5
        x = input.multiply(alpha_norm.unsqueeze(1))
        # x = input

        # self.concept_mask = self.alpha >= self.alpha.topk(self.max_complexity)[0][:, -1].unsqueeze(1)
        # # x = input.repeat(2, 1, 1).permute(0, 2, 1)
        # # x[~self.concept_mask] = 0
        # # x = x.permute(0, 2, 1)

        # compute linear map
        x = x.matmul(self.weight.permute(0, 2, 1)) + self.bias
        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_classes={}'.format(
            self.in_features, self.out_features, self.n_classes
        )


class LinearIndependent(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, n_classes: int,
                 top: bool = False, bias: bool = True) -> None:
        super(LinearIndependent, self).__init__()
        self.n_classes = n_classes
        self.top = top
        self.conceptizator = Conceptizator('identity_bool')
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
        self.conceptizator.concepts = input
        x = input.matmul(self.weight.permute(0, 2, 1)) + self.bias
        if self.top:
            x = x.view(self.n_classes, -1).t()
            self.conceptizator.concepts = x
        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_classes={}, top={}'.format(
            self.in_features, self.out_features, self.n_classes, self.top
        )


if __name__ == '__main__':
    data = torch.rand((10, 5))
    layer = ConceptAware(5, 4, 2)
    out = layer(data)
    print(out.shape)
    layer2 = LinearIndependent(4, 3, 2)
    out2 = layer2(out)
    print(out2.shape)
    layer2 = LinearIndependent(3, 1, 2)
    out3 = layer2(out2)
    print(out3.shape)
    print(out3)
