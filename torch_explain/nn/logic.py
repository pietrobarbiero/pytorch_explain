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
            self.gamma = None
            self.alpha = None #Parameter(torch.ones((n_classes, in_features)))
            # self.weight_att = Parameter(torch.Tensor(n_classes, n_heads, 1, in_features))
            # if bias:
            #     self.bias_att = Parameter(torch.Tensor(n_classes, n_heads, 1, 1))
            # else:
            #     self.register_parameter('bias_att', None)
            self.weight = Parameter(torch.Tensor(n_classes, out_features, in_features))
            if bias:
                self.bias = Parameter(torch.Tensor(n_classes, 1, out_features))
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

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        self.conceptizator.concepts = input
        x = input
        if self.shrink:
            self.gamma = self.weight.norm(dim=1)
            self.alpha = torch.softmax(self.gamma, dim=1)
            self.beta = self.alpha / self.alpha.max()
            x = input.multiply(self.beta.unsqueeze(1))
            x = x.matmul(self.weight.permute(0, 2, 1)) + self.bias
        else:
            x = x.matmul(self.weight.permute(0, 2, 1)) + self.bias
        if self.top:
            x = x.view(self.n_classes, -1).t()
            self.conceptizator.concepts = x
        return x

    def forward2(self, input: Tensor) -> Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        self.conceptizator.concepts = input
        # x = input
        if self.shrink:
            x = input.unsqueeze(0).matmul(self.weight.permute(0, 1, 3, 2)) + self.bias
            x = x.mean(dim=1)
            gamma = self.weight.norm(dim=1).norm(dim=1)
            alpha = torch.softmax(gamma, dim=1)
            x = x.multiply(alpha.unsqueeze(2)).mean(dim=1)
        #     input = input.unsqueeze(0)
        #     x = torch.tanh(input.multiply(self.weight_att) + self.bias_att)
        #     gamma = self.weight_att.norm(dim=1)
        #     alpha = torch.softmax(gamma, dim=2)
        #     x = x.multiply(alpha.unsqueeze(1)).mean(dim=1)
        # x = x.matmul(self.weight.permute(0, 2, 1)) + self.bias
        if self.top:
            x = x.view(self.n_classes, -1).t()
            self.conceptizator.concepts = x
        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_classes={}, shrink={}, top={}'.format(
            self.in_features, self.out_features, self.n_classes, self.shrink, self.top
        )


class Sequential(torch.nn.Sequential):
    def forward(self, input, train=True):
        x, y = input
        for module in self:
            if isinstance(module, Attention) and train:
                x = module(x, y)
            else:
                x = module(x)
        return x


class Attention(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, n_classes: int, first: bool = False) -> None:
        super(Attention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.first = first
        self.d_key_query = 5
        self.d_value = 1
        self.w_key = Parameter(torch.Tensor(n_classes, 1, 1, self.d_key_query))
        self.w_query = Parameter(torch.Tensor(n_classes, 1, self.d_key_query))
        self.w_value = Parameter(torch.Tensor(n_classes, 1, self.d_value))
        self.weight = Parameter(torch.Tensor(n_classes, 1, in_features, out_features))
        self.bias = Parameter(torch.Tensor(n_classes, 1, out_features))
        self.attn_scores_softmax = None
        self.reset_parameters()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.first:
            x = x.unsqueeze(0)
        y = y.t()
        keys = x.unsqueeze(-1) @ self.w_key
        querys = y.unsqueeze(-1) @ self.w_query
        values = x * self.w_value
        attn_scores = keys @ querys.unsqueeze(-1)
        attn_scores_avg = attn_scores.mean(dim=1).permute(0, 2, 1)
        self.attn_scores_softmax = torch.softmax(attn_scores_avg, dim=-1)
        weighted_values = values * (self.attn_scores_softmax / self.attn_scores_softmax.max(dim=2)[0].unsqueeze(-1))
        if self.out_features > 1:
            y = (weighted_values.unsqueeze(-2) @ self.weight).squeeze() + self.bias
        else:
            y = (weighted_values.unsqueeze(-2) @ self.weight).squeeze(-1) + self.bias
        # print(y.shape)
        return y

    def reset_parameters(self) -> None:
        # init.kaiming_uniform_(self.w_key, a=math.sqrt(5))
        # init.kaiming_uniform_(self.w_query, a=math.sqrt(5))
        # init.kaiming_uniform_(self.w_value, a=math.sqrt(5))
        init.kaiming_uniform_(self.w_key, a=math.sqrt(5))
        init.kaiming_uniform_(self.w_query, a=math.sqrt(5))
        init.kaiming_uniform_(self.w_value, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    # def extra_repr(self) -> str:
    #     return 'in_features={}, out_features={}, n_classes={}, shrink={}, top={}'.format(
    #         self.in_features, self.out_features, self.n_classes, self.shrink, self.top
    #     )


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
