import abc
import torch
from torch.nn import functional as F


class Logic:
    @abc.abstractmethod
    def update(self):
        raise NotImplementedError

    @abc.abstractmethod
    def conj(self, a, dim=1):
        raise NotImplementedError

    @abc.abstractmethod
    def disj(self, a, dim=1):
        raise NotImplementedError

    def conj_pair(self, a, b):
        raise NotImplementedError

    def disj_pair(self, a, b):
        raise NotImplementedError

    def iff_pair(self, a, b):
        raise NotImplementedError

    @abc.abstractmethod
    def neg(self, a):
        raise NotImplementedError


class ProductTNorm(Logic):
    def __init__(self):
        super(ProductTNorm, self).__init__()
        self.current_truth = torch.tensor(1)
        self.current_false = torch.tensor(0)

    def update(self):
        pass

    def conj(self, a, dim=1):
        return torch.prod(a, dim=dim, keepdim=True)

    def conj_pair(self, a, b):
        return a * b

    def disj(self, a, dim=1):
        return 1 - torch.prod(1 - a, dim=dim, keepdim=True)

    def disj_pair(self, a, b):
        return a + b - a * b

    def iff_pair(self, a, b):
        return self.conj_pair(self.disj_pair(self.neg(a), b), self.disj_pair(a, self.neg(b)))

    def neg(self, a):
        return 1 - a

    def predict_proba(self, a):
        return a.squeeze(-1)


class GodelTNorm(Logic):
    def __init__(self):
        super(GodelTNorm, self).__init__()
        self.current_truth = 1
        self.current_false = 0

    def update(self):
        pass

    def conj(self, a,dim=1):
        return torch.min(a, dim=dim, keepdim=True)[0]

    def disj(self, a, dim=1):
        return torch.max(a, dim=dim, keepdim=True)[0]

    def conj_pair(self, a, b):
        return torch.minimum(a, b)

    def disj_pair(self, a, b):
        return torch.maximum(a, b)

    def iff_pair(self, a, b):
        return self.conj_pair(self.disj_pair(self.neg(a), b), self.disj_pair(a, self.neg(b)))

    def neg(self, a):
        return 1 - a

    def predict_proba(self, a):
        return a.squeeze(-1)
