import abc
import torch


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

class VectorLogic(Logic):
    def __init__(self, emb_size):
        super(VectorLogic, self).__init__()
        self.emb_size = emb_size
        self.truth = torch.nn.Parameter(torch.randn(emb_size, 1), requires_grad=False)
        self.false = torch.nn.Parameter(torch.randn(emb_size, 1), requires_grad=False)
        self.update()
        # truth = torch.FloatTensor([[1], [0]])
        # false = torch.FloatTensor([[0], [1]])

    def update(self):
        # make reference vectors orthonormal
        truth_norm = torch.norm(self.truth.clone())
        truth = self.truth.clone() / truth_norm
        truth_false_proj = truth.T.matmul(self.false)
        false = self.false - truth_false_proj * truth
        false_norm = torch.norm(false)
        false = false / false_norm

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
