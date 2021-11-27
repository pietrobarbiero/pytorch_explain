#%%
from typing import Union, Callable, Optional

import torch
from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm
from torch.nn.modules.transformer import _get_activation_fn
from torchvision.models import resnet18
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from tqdm.autonotebook import tqdm
import numpy as np

#%%

model = resnet18(num_classes=10)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=20, bias=True)

#%%

train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor())
test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())

#%%

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)

#%%

from torch import Tensor

def context(embedding: Tensor) -> Tensor:
    return embedding / torch.norm(embedding, p=2, dim=-1).unsqueeze(-1)

def logprobs(embedding: Tensor) -> Tensor:
    return torch.norm(embedding, p=2, dim=-1)

def semantics(embedding: Tensor) -> Tensor:
    return torch.exp(-torch.norm(embedding, p=2, dim=-1))


class ConceptEmbeddings(nn.Linear):
    def __init__(self, in_features: int, out_features: int, emb_size: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConceptEmbeddings, self).__init__(in_features, out_features, bias, device, dtype)
        self.weight = nn.Parameter(torch.empty((out_features, in_features, emb_size), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, emb_size, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        return (input @ self.weight).permute(1, 0, 2) + self.bias

input = torch.randn(128, 20)
c2e = ConceptEmbeddings(20, 10, 2)
embedding = c2e(input)
ct = context(embedding)
lp = logprobs(embedding)
sm = semantics(embedding)
torch.max(sm)
torch.min(sm)

class ResNetMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = ConceptEmbeddings(in_features=512, out_features=10, emb_size=2, bias=True)
        self.loss = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        embedding = self(x)
        logits = semantics(embedding)
        loss = self.loss(logits, torch.nn.functional.one_hot(y, num_classes=10).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)

#%%

# model = ResNetMNIST()
# logits = model(next(iter(train_dl))[0]).reshape(-1, 10, 2)
# logits.norm(dim=2)

#%%

model = ResNetMNIST()
trainer = pl.Trainer(gpus=1, max_epochs=1)
trainer.fit(model, train_dl)

#%%

def get_prediction(x, model: pl.LightningModule):
    model.freeze() # prepares model for predicting
    embedding = model(x)
    logits = semantics(embedding)
    predicted_class = torch.argmax(logits, dim=1)
    return predicted_class, logits

true_y, pred_y = [], []
for batch in tqdm(iter(test_dl), total=len(test_dl)):
    x, y = batch
    true_y.extend(y)
    preds, probs = get_prediction(x, model)
    pred_y.extend(preds.cpu())

print(classification_report(true_y, pred_y, digits=3))

#%%

embedding = model(x)
sm = torch.round(semantics(embedding) + 1)
ct = context(embedding)
norm_embedding = ct * sm.unsqueeze(-1)
norm_embedding2 = norm_embedding.reshape(norm_embedding.shape[0] * 10, 2)
numbs = torch.IntTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).repeat(64)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.scatterplot(norm_embedding2[:, 0], norm_embedding2[:, 1], hue=numbs)
plt.show()



## sum training set

def get_prediction(x, model: pl.LightningModule):
    model.freeze() # prepares model for predicting
    embedding = model(x)
    logits = semantics(embedding)
    predicted_class = torch.argmax(logits, dim=1)
    return predicted_class, logits

c_list, y_list = [], []
for batch in tqdm(iter(test_dl), total=len(test_dl)):
    x, y = batch
    preds, probs = get_prediction(x, model)
    ok_idx = y.eq(preds)
    n_ok = ok_idx.sum() // 2
    ok_preds = preds[ok_idx][:n_ok]
    ok_preds_rev = preds[ok_idx].flip(0)[:n_ok]
    y_sum = ok_preds + ok_preds_rev
    y_list.extend(y_sum)

    x_ok = x[ok_idx][:n_ok]
    embedding = model(x_ok)
    sm = torch.round(semantics(embedding) + 1)
    ct = context(embedding)
    norm_embedding = sm.unsqueeze(-1) * ct
    # norm_embedding2 = norm_embedding.reshape(norm_embedding.shape[0] * 10, 2)


    x_ok = x[ok_idx].flip(0)[:n_ok]
    embedding = model(x_ok)
    sm = torch.round(semantics(embedding) + 1)
    ct = context(embedding)
    norm_embedding2 = ct * sm.unsqueeze(-1)
    # norm_embedding3 = norm_embedding.reshape(norm_embedding.shape[0] * 10, 2)
    c_list.extend(torch.concat([norm_embedding, norm_embedding2], dim=1))

c = torch.stack(c_list).float()
y = torch.nn.functional.one_hot(torch.stack(y_list)).float()

c2y_model = nn.Sequential(*[
    nn.Linear(20, 20),
    nn.LeakyReLU(),
    nn.Linear(20, 19),
])
loss = nn.BCELoss()
opt = torch.optim.Adam(c2y_model.parameters(), lr=0.005)
for epoch in range(4000):
    opt.zero_grad()
    emb_pred = c2y_model(c.permute(0, 2, 1)).permute(0, 2, 1)
    y_pred = semantics(emb_pred)
    ls = loss(y_pred, y)
    ls.backward()
    opt.step()

    # compute accuracy
    if epoch % 100 == 0:
        accuracy = (y_pred > 0.5).eq(y).sum().item() / (y.size(0) * y.size(1))
        print(f'Epoch {epoch}: loss {ls:.4f} train accuracy: {accuracy:.4f}')

embedding = c2y_model(c.permute(0, 2, 1)).permute(0, 2, 1).detach()
sm_a = c[:, :10].norm(dim=-1)
sm_b = c[:, 10:].norm(dim=-1)
sm = torch.round(semantics(embedding) + 1)
ct = context(embedding)
norm_embedding2 = ct * sm.unsqueeze(-1)

na = torch.argmax(sm_a[0])
nb = torch.argmax(sm_b[0])
nab = torch.argmax(sm[0])

import pylab as p

sns.set_style("whitegrid")

plt.figure(figsize=[6, 6])
plt.title(f'is{nab}(img1,img2) <- is{na}(img1), is{nb}(img2)')
sns.scatterplot(norm_embedding2[0, :, 0], norm_embedding2[0, :, 1], label=f'isX(img1,img2), X!=10')
sns.scatterplot([norm_embedding2[0, nab, 0]], [norm_embedding2[0, nab, 1]], label=f'is{nab}(img1,img2)')
sns.scatterplot([c[0, na, 0]], [c[0, na, 1]], label=f'is{na}(img1)')
sns.scatterplot([c[0, 10+nb, 0]], [c[0, 10+nb, 1]], label=f'is{nb}(img2)')

p.arrow(0, 0, norm_embedding2[0, nab, 0]-0.1, norm_embedding2[0, nab, 1]+0.07, fc="k", ec="k", head_width=0.05, head_length=0.1 )
p.arrow(0, 0, c[0, na, 0]-0.06, c[0, na, 1]+0.11, fc="k", ec="k",head_width=0.05, head_length=0.1 )
p.arrow(0, 0, c[0, 10+nb, 0]-0.11, c[0, 10+nb, 1]+0.05, fc="k", ec="k",head_width=0.05, head_length=0.1 )

plt.xlabel('Emb dim #1')
plt.ylabel('Emb dim #2')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.savefig('mnist_sum.png')
plt.savefig('mnist_sum.pdf')
plt.show()



class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)