import math
from typing import Optional, Tuple
import torch
from torch.nn import Linear, Parameter, Module, MultiheadAttention, Module, init
from torch import Tensor
from torch.nn import functional as F


class ConceptEmbeddings(Linear):
    def __init__(self, in_features: int, out_features: int, emb_size: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConceptEmbeddings, self).__init__(in_features, out_features, bias, device, dtype)
        self.weight = Parameter(torch.empty((out_features, in_features, emb_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, emb_size, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        h = (input @ self.weight).permute(1, 0, 2) + self.bias
        # return embedding_to_nesyemb(h)
        return h.permute(0, 2, 1)


class Flatten2Emb(Module):

    def __init__(self) -> None:
        super(Flatten2Emb, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input.reshape(input.shape[0], input.shape[1], -1).permute(0, 2, 1)


class SoftmaxTemp(Module):
    __constants__ = ['dim']
    dim: Optional[int]
    temperature: float

    def __init__(self, dim: Optional[int] = None, temperature: float = 1) -> None:
        super(SoftmaxTemp, self).__init__()
        self.dim = dim
        self.temperature = temperature
        self.alpha = None
        self.alpha_norm = None

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        self.alpha = torch.exp(input / self.temperature) / torch.sum(torch.exp(input / self.temperature),
                                                                     dim=self.dim, keepdim=True)
        self.alpha_norm = self.alpha / self.alpha.max(dim=self.dim)[0].unsqueeze(1)
        return self.alpha

    def extra_repr(self) -> str:
        return 'dim={dim}, temperature={temperature}'.format(dim=self.dim, temperature=self.temperature)


class NeSyLayer(Module):
    def __init__(self, embed_dim: int, in_concepts: int, h_concepts: int, n_classes: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super(NeSyLayer, self).__init__()
        self.emb_size = embed_dim
        self.in_concepts = in_concepts
        self.h_concepts = h_concepts
        self.n_classes = n_classes
        self.bias = bias
        self.attn_layer = NeSyGate(embed_dim, n_classes, bias, device, dtype)
        self.linear1 = Linear(in_concepts, h_concepts, bias, device, dtype)
        self.linear2 = Linear(h_concepts, 1, bias, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        h = self.attn_layer(input)
        h = self.linear1(h)
        h = F.leaky_relu(h)
        h = self.linear2(h).squeeze(-1)
        out = embedding_to_nesyemb(h)
        return out

    def extra_repr(self) -> str:
        return 'emb_size={}, in_concepts={}, bias={}'.format(
            self.emb_size, self.in_concepts, self.bias is not None
        )


class NeSyGate(Module):
    def __init__(self, embed_dim: int, out_concepts: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super(NeSyGate, self).__init__()
        self.emb_size = embed_dim
        self.out_concepts = out_concepts
        self.bias = bias
        self.w_key = Linear(embed_dim, embed_dim, bias, device, dtype)
        self.w_query = Linear(embed_dim, out_concepts, bias, device, dtype)
        self.attn_scores = None

    def forward(self, input: Tensor) -> Tensor:
        key = self.w_key(input)
        attn_scores = self.w_query(key)
        self.attn_scores = F.softmax(attn_scores.mean(dim=0), dim=0) #F.relu(attn_score).mean(dim=0)
        value_scored = (input.unsqueeze(2) * self.attn_scores.unsqueeze(0).unsqueeze(-1)).permute(0, 2, 3, 1)
        return value_scored

    def extra_repr(self) -> str:
        return 'emb_size={}, out_concepts={}, bias={}'.format(
            self.emb_size, self.out_concepts, self.bias is not None
        )


def embedding_to_nesyemb(embedding: Tensor) -> Tensor:
    embedding = embedding.permute(0, 2, 1)
    embedding_norm = embedding.norm(dim=-1).unsqueeze(dim=-1)
    return embedding / embedding_norm * (torch.exp(-embedding_norm))


def context(embedding: Tensor) -> Tensor:
    return embedding / torch.norm(embedding, p=2, dim=-1).unsqueeze(-1)


def logprobs(embedding: Tensor) -> Tensor:
    return torch.norm(embedding, p=2, dim=-1)


def semantics(embedding: Tensor) -> Tensor:
    return torch.norm(embedding, dim=-1) #torch.exp(-torch.norm(embedding, p=2, dim=-1))
    # return torch.norm(embedding, dim=-1).log().sigmoid()
    # return torch.exp(-torch.norm(embedding, dim=-1))


def to_boolean(embedding: Tensor, true_norm: float = 0, false_norm: float = 1) -> Tensor:
    sm = torch.round(semantics(embedding))
    sm[sm != 0] = false_norm
    sm[sm == 0] = true_norm
    ct = context(embedding)
    return ct * sm.unsqueeze(-1)


if __name__ == '__main__':
    # Self-supervised concepts
    from torch_explain.nn.functional.loss import entropy_logic_loss
    n_samples = 64
    n_filters = 32
    emb_size = 3*3
    input = torch.randn(n_samples, 1, 5, 5)
    x2c = torch.nn.Sequential(
        torch.nn.Conv2d(1, n_filters, 5, 1, 1),
        Flatten2Emb(),
    )
    scoring = torch.nn.Sequential(
        Linear(emb_size, 1),
        SoftmaxTemp(dim=1, temperature=1),
    )
    concept_emb = x2c(input)
    concept_scores = scoring(concept_emb.permute(0, 2, 1)).squeeze(-1)
    loss = entropy_logic_loss(scoring)
    print(torch.sum(concept_scores, dim=1))
    print(torch.sum(concept_scores, dim=1).shape)
    print("Loss: ", loss)

    # Supervised concepts
    n_samples = 100
    n_features = 50
    n_concepts = 5
    emb_size = 3
    n_classes = 2

    x = torch.randn(n_samples, n_features)

    model = torch.nn.Sequential(*[
        ConceptEmbeddings(n_features, n_concepts, emb_size),
        NeSyGate(emb_size, n_classes),
        torch.nn.Linear(n_concepts, 10),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 1),
    ])
    y_emb = model(x).squeeze(-1)
    print(y_emb.shape)

    model2 = torch.nn.Sequential(*[
        ConceptEmbeddings(n_features, n_concepts, emb_size),
        NeSyLayer(emb_size, n_concepts, 3, 10),
        NeSyLayer(emb_size, 10, 3, n_classes),
    ])
    y_emb = model2(x)
    print(y_emb.shape)
    print(semantics(y_emb))
