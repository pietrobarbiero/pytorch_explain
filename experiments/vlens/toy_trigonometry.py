import os
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.nn import Conv2d, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, Sequential, LeakyReLU, Linear
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, inception_v3
import torch_explain as te
from experiments.data.load_datasets import load_cub_full
from torch_explain.nn import ConceptEmbeddings, context, semantics, to_boolean
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


def main():
    seed_everything(42)
    x = np.random.randn(1000, 2)
    c = np.array([np.sin(x[:, 0])>0.5, np.cos(x[:, 0])>0.5]).T
    y = np.tan(x[:, 0])>0.5
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)
    # y = one_hot(torch.LongTensor(y)).float()

    train_data = TensorDataset(x, c, y)

    train_dl = DataLoader(train_data, batch_size=1000)
    # test_dl = DataLoader(test_data, batch_size=batch_size)

    model = TrigoNetEmb()
    # c_pred, y_pred = model(next(iter(train_dl))[0])
    trainer = pl.Trainer(gpus=0, max_epochs=2000)
    trainer.fit(model, train_dl)
    #
    # result_dir = './results/trigonometry/'
    # os.makedirs(result_dir, exist_ok=True)
    # trainer.save_checkpoint(os.path.join(result_dir, 'model.pt'))

    return


class TrigoNetEmb(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.x2c_model = Sequential(*[
            Linear(2, 10),
            LeakyReLU(),
            ConceptEmbeddings(in_features=10, out_features=2, emb_size=2, bias=True),
        ])
        self.c2y_model = Sequential(*[
            Linear(2, 10),
            LeakyReLU(),
            Linear(10, 1),
        ])
        self.loss = BCELoss()

    def forward(self, x):
        c = self.x2c_model(x)
        c_bool = to_boolean(c, true_norm=1, false_norm=2)
        y = self.c2y_model(c_bool.permute(0, 2, 1)).permute(0, 2, 1)
        return c, y

    def training_step(self, batch, batch_no):
        x, c, y = batch
        c_emb, y_emb = self(x)
        c_logits = semantics(c_emb)
        y_logits = semantics(y_emb)
        y_logits = y_logits.reshape(-1)
        loss = self.loss(c_logits, c) + self.loss(y_logits, y)

        # compute accuracy
        c_pred = c_logits.reshape(-1).cpu().detach() > 0.5
        y_pred = y_logits.reshape(-1).cpu().detach() > 0.5
        c_true = c.reshape(-1).cpu().detach()
        y_true = y.reshape(-1).cpu().detach()
        c_accuracy = accuracy_score(c_true, c_pred)
        y_accuracy = accuracy_score(y_true, y_pred)
        # c_f1 = f1_score(c_true, c_pred)
        # y_f1 = f1_score(y_true, y_pred)
        print(f'Loss {loss:.4f}, '
              # f'c_f1: {c_f1:.4f}, y_f1: {y_f1:.4f}, '
              f'c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}, ')

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


class TrigoNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.x2c_model = Sequential(*[
            Linear(2, 10),
            LeakyReLU(),
            Linear(10, 2),
        ])
        self.c2y_model = Sequential(*[
            Linear(2, 10),
            LeakyReLU(),
            Linear(10, 1),
        ])
        self.loss = BCEWithLogitsLoss()

    def forward(self, x):
        c = self.x2c_model(x)
        y = self.c2y_model((c>0.5).float())
        return c, y

    def training_step(self, batch, batch_no):
        x, c, y = batch
        c_logits, y_logits = self(x)
        y_logits = y_logits.reshape(-1)
        loss = self.loss(c_logits, c) + self.loss(y_logits, y)

        # compute accuracy
        c_pred = c_logits.cpu().detach() > 0.5
        y_pred = y_logits.cpu().detach() > 0.5
        c_true = c.cpu().detach()
        y_true = y.cpu().detach()
        c_accuracy = accuracy_score(c_true, c_pred)
        y_accuracy = accuracy_score(y_true, y_pred)
        # c_f1 = f1_score(c_true, c_pred)
        # y_f1 = f1_score(y_true, y_pred)
        print(f'Loss {loss:.4f}, '
              # f'c_f1: {c_f1:.4f}, y_f1: {y_f1:.4f}, '
              f'c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}, ')

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


if __name__ == '__main__':
    main()
