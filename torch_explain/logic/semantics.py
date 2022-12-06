import abc
import torch


class Logic:

    @abc.abstractmethod
    def conj(self, a):
        raise NotImplementedError

    @abc.abstractmethod
    def disj(self, a):
        raise NotImplementedError

    @abc.abstractmethod
    def neg(self, a):
        raise NotImplementedError


class ProductTNorm(Logic):

    def conj(self, a):
        return torch.prod(a, dim=1, keepdim=True)

    def disj(self, a):
        return 1 - torch.prod(1 - a, dim=1, keepdim=True)

    def neg(self, a):
        return 1 - a
