import torch
from torch import Tensor
from torch.nn import Module


class Conceptizator(Module):
    """Saves the encoding for input concepts.
    """

    def __init__(self, activation: str = 'sigmoid') -> None:
        super(Conceptizator, self).__init__()
        self.concepts = None
        self.activation_name = activation
        self.activation = torch.sigmoid
        # if self.activation_name == 'sigmoid':
        #     self.activation = torch.sigmoid
        #     self.threshold = 0.5
        # if self.activation_name == 'relu':
        #     self.activation = torch.relu
        #     self.threshold = 0.
        # if self.activation_name == 'leaky_relu':
        #     self.activation = torch.nn.functional.leaky_relu
        #     self.threshold = 0.
        # if self.activation_name == 'identity':
        #     self.activation = identity
        #     self.threshold = 0.
        if self.activation_name == 'identity_bool':
            self.activation = identity
            self.threshold = 0.5

    def forward(self, input: Tensor) -> Tensor:
        self.concepts = self.activation(input)
        return self.concepts

    def extra_repr(self) -> str:
        return 'activation={}, threshold={}'.format(self.activation_name, self.threshold)


def identity(x):
    return x
