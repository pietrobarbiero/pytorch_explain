import abc
import torch
from torch.nn import functional as F


class Logic:
    @abc.abstractmethod
    def update(self):
        raise NotImplementedError

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

    def update(self):
        pass

    def conj(self, a):
        return torch.prod(a, dim=1, keepdim=True)

    def disj(self, a):
        return 1 - torch.prod(1 - a, dim=1, keepdim=True)

    def neg(self, a):
        return 1 - a


class GodelTNorm(Logic):

    def update(self):
        pass

    def conj(self, a):
        return torch.min(a, dim=1, keepdim=True)

    def disj(self, a):
        return torch.max(a, dim=1, keepdim=True)

    def neg(self, a):
        return 1 - a

class VectorLogic(Logic, torch.nn.Module):
    def __init__(self, emb_size):
        super(VectorLogic, self).__init__()
        self.emb_size = emb_size
        self._truth = torch.nn.Parameter(torch.randn(emb_size, 1), requires_grad=True)
        self._false = torch.randn(self.truth.shape)
        # self.false = torch.nn.Parameter(torch.randn(emb_size, 1), requires_grad=True)
        torch.nn.init.normal_(self.truth)
        # torch.nn.init.normal_(self.false)
        # truth = torch.FloatTensor([[1], [0]])
        # false = torch.FloatTensor([[0], [1]])
        self.update()

    @property
    def truth(self):
        return F.normalize(self._truth, p=2, dim=0)

    @property
    def false(self):
        truth = self.truth
        truth_false_proj = truth.T.matmul(self._false)
        false = F.normalize(self._false - truth_false_proj * truth, p=2, dim=0)
        return false

    def update(self):
        truth = self.truth
        false = self.false

        tt = torch.kron(truth, truth).T
        tf = torch.kron(truth, false).T
        ft = torch.kron(false, truth).T
        ff = torch.kron(false, false).T

        self.current_truth = truth
        self.current_false = false
        self.negation = false.matmul(truth.T) + truth.matmul(false.T)
        self.conjunction = truth.matmul(tt) + false.matmul(tf) + false.matmul(ft) + false.matmul(ff)
        self.disjunction = truth.matmul(tt) + truth.matmul(tf) + truth.matmul(ft) + false.matmul(ff)

        # check axioms
        eps = 1e-5
        assert torch.matmul(truth.T, false).squeeze() < eps # orthonormality
        assert torch.all(self.negation.matmul(self.negation.matmul(truth)) - truth < eps) # involution
        assert torch.all(self.negation.matmul(self.negation.matmul(false)) - false < eps)
        assert torch.all(self.conjunction.matmul(torch.kron(truth, truth)) - truth < eps) # conjunction
        assert torch.all(self.conjunction.matmul(torch.kron(truth, false)) - false < eps)
        assert torch.all(self.conjunction.matmul(torch.kron(false, truth)) - false < eps)
        assert torch.all(self.conjunction.matmul(torch.kron(false, false)) - false < eps)
        assert torch.all(self.disjunction.matmul(torch.kron(false, false)) - false < eps) # disjunction
        assert torch.all(self.disjunction.matmul(torch.kron(truth, false)) - truth < eps)
        assert torch.all(self.disjunction.matmul(torch.kron(false, truth)) - truth < eps)
        assert torch.all(self.disjunction.matmul(torch.kron(truth, truth)) - truth < eps)

    def _compose(self, a1, a2):
        return torch.vstack([torch.kron(a1[r], a2[r]) for r in range(a1.shape[0])]).T

    def conj(self, a):
        return self.conjunction.matmul(self._compose(a[:, 0], a[:, 1])).T.unsqueeze(1)

    def disj(self, a):
        return self.disjunction.matmul(self._compose(a[:, 0], a[:, 1])).T.unsqueeze(1)

    def neg(self, a):
        return self.negation.matmul(a[:, 0].T).T.unsqueeze(1)
