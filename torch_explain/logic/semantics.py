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

    def disj(self, a, dim=1):
        return 1 - torch.prod(1 - a, dim=dim, keepdim=True)

    def neg(self, a):
        return 1 - a

    def predict_proba(self, a):
        return a.squeeze(-1)

class SumProductSemiring(Logic):
    def __init__(self):
        super(SumProductSemiring, self).__init__()
        self.current_truth = 1
        self.current_false = 0

    def update(self):
        pass

    def conj(self, a, dim=1):
        return torch.prod(a, dim=dim, keepdim=True)

    def disj(self, a, dim=1):
        return torch.sum(a, dim=dim, keepdim=True)

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

    def neg(self, a):
        return 1 - a

    def predict_proba(self, a):
        return a.squeeze(-1)


class VectorLogic(Logic, torch.nn.Module):
    def __init__(self, emb_size, cuda = True):
        super(VectorLogic, self).__init__()
        self.emb_size = emb_size
        self._truth = torch.nn.Parameter(torch.randn(emb_size, 1), requires_grad=False)  # TODO: check if we really need to train logic
        self._false = torch.randn(self.truth.shape)
        torch.nn.init.normal_(self.truth)
        self._check_axioms()
        self.cuda = cuda

    def _check_axioms(self):
        self.update()
        if self.cuda:
            truth = self.truth.cuda()
            false = self.false.cuda()
        else:
            truth = self.truth.cuda()
            false = self.false.cuda()
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

    @property
    def truth(self):
        return F.normalize(self._truth, p=2, dim=0)

    @property
    def false(self):
        truth = self.truth
        truth_false_proj = truth.T.matmul(self._false)
        false = F.normalize(self._false - truth_false_proj * truth, p=2, dim=0)
        return false

    def get_truth_from_embeddings(self, embeddings):
        return F.normalize(embeddings, p=2, dim=0)

    def get_false_from_truth(self, truth):
        truth_false_proj = truth.matmul(self._false)
        false = F.normalize(self._false.squeeze(-1).unsqueeze(0) - truth_false_proj * truth, p=2, dim=0)
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
        if self.cuda:
            self.current_truth = self.current_truth.cuda()
            self.current_false = self.current_false.cuda()
            self.negation = self.negation.cuda()
            self.conjunction = self.conjunction.cuda()
            self.disjunction = self.disjunction.cuda()

    def _compose(self, a1, a2):
        return torch.vstack([torch.kron(a1[r], a2[r]) for r in range(a1.shape[0])]).T   # TODO: find smarter way

    def conj(self, a, dim=1):
        s = torch.unbind(a, dim=dim)
        r = s[0]
        if len(s) > 1:
            for j in range(1, len(s)):
                r = self.conjunction.matmul(self._compose(r, s[j])).T.unsqueeze(1)
        return r
    def disj(self, a, dim=1):
        s = torch.unbind(a, dim=dim)
        r = s[0].unsqueeze(1)
        if len(s) > 1:
            for j in range(1, len(s)):
                r = self.disjunction.matmul(self._compose(r, s[j])).T.unsqueeze(1)
        return r


    def neg(self, a):
        return self.negation.matmul(a[:, 0].T).T.unsqueeze(1)

    def predict_proba(self, a):
        return torch.clamp(a.matmul(self.current_truth), 0, 1).squeeze(-1)
