import math
from typing import Optional, Tuple
import torch
from torch.nn import Linear, Parameter, Module, MultiheadAttention
from torch import Tensor


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
        return (input @ self.weight).permute(1, 0, 2) + self.bias


class NeSyAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        super(NeSyAttention, self).__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                                       kdim, vdim, batch_first, device, dtype)
        self.alpha = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        attn_output, attn_output_weights = self.attn(query, key, value, key_padding_mask, True, attn_mask)
        self.alpha = attn_output_weights.mean(dim=0) / attn_output_weights.mean(dim=0).max(dim=-1)[0].unsqueeze(-1)
        nesy_attn_output = embedding_to_nesyemb(attn_output)
        return nesy_attn_output, attn_output_weights


def embedding_to_nesyemb(embedding: Tensor) -> Tensor:
    embedding_norm = embedding.norm(dim=-1).unsqueeze(dim=-1)
    return embedding / embedding_norm * (torch.exp(-embedding_norm) + 1)


def context(embedding: Tensor) -> Tensor:
    return embedding / torch.norm(embedding, p=2, dim=-1).unsqueeze(-1)


def logprobs(embedding: Tensor) -> Tensor:
    return torch.norm(embedding, p=2, dim=-1)


def semantics(embedding: Tensor) -> Tensor:
    return torch.exp(-torch.norm(embedding, p=2, dim=-1))


def to_boolean(embedding: Tensor, true_norm: float = 0, false_norm: float = 1) -> Tensor:
    sm = torch.round(semantics(embedding))
    sm[sm != 0] = false_norm
    sm[sm == 0] = true_norm
    ct = context(embedding)
    return ct * sm.unsqueeze(-1)


if __name__ == '__main__':
    import torch
    from torch.nn import MultiheadAttention, TransformerEncoder, TransformerDecoderLayer
    from sklearn.metrics import accuracy_score, f1_score

    n_samples = 1000
    n_concepts = 5
    emb_size = 20
    n_classes = 2
    x = torch.randn(n_concepts, n_samples, emb_size)
    y = torch.randn(n_classes, n_samples, emb_size)
    x[0] = y[0] = torch.randn(n_samples, emb_size)
    y[1] = x[2] * x[3]
    (y[0]>0).float().mean()
    # y = y / y.norm(dim=-1).unsqueeze(-1)
    # y[0] = y[0] * (((x[0].mean(dim=-1))>0)+1).unsqueeze(-1)
    # y[1] = y[1] * ((((x[2].mean(dim=-1)>0).float() * (x[3].mean(dim=-1)>0).float())>0)+1).unsqueeze(-1)
    x[1] = x[4] = 0
    y_ctx = context(y)
    y[0] = y_ctx[0] * ((y[0]>0) + 1)#.unsqueeze(-1)
    y[1] = y_ctx[1] * ((y[1]>0) + 1)#.unsqueeze(-1)

    layers = [
        NeSyAttention(emb_size, 5),
        NeSyAttention(emb_size, 5),
        # MultiheadAttention(emb_size, 5),
        # MultiheadAttention(emb_size, 5),
    ]
    model = torch.nn.Sequential(*layers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form = torch.nn.BCELoss()

    for epoch in range(2000):
        optimizer.zero_grad()
        loss_alpha = 0
        h = x
        for i, mod in enumerate(layers):
            if i == 0:
                h, w = mod(h, h, h)
            elif i == 1:
                y_pred, w = mod(y, h, h)

            # loss -= torch.sum(w.reshape(-1, h.shape[0]) * torch.log(w.reshape(-1, h.shape[0]))) / h.shape[1]
            loss_alpha -= torch.sum(w.mean(dim=0) * torch.log(w.mean(dim=0))) / w[0].ravel().shape[0]

        # y_pred_norm = semantics(y_pred)
        # y_norm = semantics(y)
        y_pred_sem = (y_pred.norm(dim=-1)-1).ravel()
        y_sem = (y.norm(dim=-1)-1).ravel()
        loss = loss_form(y_pred_sem, y_sem) + 0.5 * loss_alpha
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 500 == 0:
            f1 = accuracy_score(y_pred_sem>0.5, y_sem>0.5)
            print(f'Epoch {epoch}: loss {loss:.4f} train f1: {f1:.4f}')

    import matplotlib.pyplot as plt
    import seaborn as sns
    h = x
    for i, mod in enumerate(layers):
        if i == 0:
            h, w = mod(h, h, h)
        elif i == 1:
            y_pred, w = mod(y, h, h)

        alpha = w.mean(dim=0) / w.mean(dim=0).max(dim=-1)[0].unsqueeze(-1)
        plt.figure()
        sns.heatmap(alpha.detach())
        plt.show()


    # import torch
    # from torch.nn import MultiheadAttention
    # from sklearn.metrics import accuracy_score
    #
    # n_samples = 100
    # n_concepts = 5
    # emb_size = 40
    # n_classes = 2
    # x = torch.randn(n_concepts, n_samples, emb_size)
    # y = torch.randn(n_classes, n_samples, emb_size)
    # y[0] = x[0]
    # y[1] = x[2] * x[3]
    # # y = y / y.norm(dim=-1).unsqueeze(-1)
    # # y[0] = y[0] * (((x[0].mean(dim=-1))>0)+1).unsqueeze(-1)
    # # y[1] = y[1] * ((((x[2].mean(dim=-1)>0).float() * (x[3].mean(dim=-1)>0).float())>0)+1).unsqueeze(-1)
    # x[1] = x[4] = 0
    #
    # mha = MultiheadAttention(emb_size, num_heads=10)
    # out, w = mha.forward(y, x, x)
    #
    # model = torch.nn.Sequential(*[mha])
    #
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # loss_form = torch.nn.MSELoss()
    #
    # for epoch in range(10000):
    #     optimizer.zero_grad()
    #     y_pred, w = mha(y, x, x)
    #     alpha = w.mean(dim=0) / w.mean(dim=0).max(dim=-1)[0].unsqueeze(-1)
    #     # y_pred_norm = y_pred.norm(dim=-1)
    #     y_pred_norm = semantics(y_pred)
    #     y_norm = semantics(y)
    #     loss = loss_form(y_pred, y) - torch.sum(w.mean(dim=0) * torch.log(w.mean(dim=0))) #w.norm(p=1/2, dim=-1).norm()
    #     loss.backward()
    #     optimizer.step()
    #
    #     # compute accuracy
    #     if epoch % 500 == 0:
    #         accuracy = accuracy_score((y_pred_norm>0.5).ravel(), (y_norm>1).ravel())
    #         print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')
    #         print(f'\t{alpha}')
    #
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.figure()
    # sns.heatmap(alpha.detach())
    # plt.show()

    # gate = ConceptGate(n_concepts, emb_size, n_classes)
    # h = gate.forward(x)
    # out = x.unsqueeze(3) * h.unsqueeze(0).unsqueeze(2)
    # print(h)
    # print(out.shape)
    # print(h.sum(0))
