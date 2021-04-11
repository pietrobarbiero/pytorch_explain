import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.utils import prune

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
