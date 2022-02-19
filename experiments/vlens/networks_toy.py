import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, normalized_mutual_info_score
from torch.nn import BCELoss, Sequential, LeakyReLU, Linear, Sigmoid
import pytorch_lightning as pl
from torch_explain.nn import ConceptEmbeddings, semantics
from torch_explain.nn.vector_logic import to_boolean, embedding_to_nesyemb


class ToyNetEmbPlane(pl.LightningModule):
    def __init__(self, n_features, n_concepts, n_tasks, emb_size):
        super().__init__()
        self.save_hyperparameters()
        self.x2c_model = Sequential(
            Linear(n_features, 10),
            LeakyReLU(),
            Linear(10, 10),
            LeakyReLU(),
            ConceptEmbeddings(in_features=10, out_features=n_concepts, emb_size=emb_size, bias=True),
        )
        self.c2y_model = Sequential(
            Linear(n_concepts, 50),
            LeakyReLU(),
            Linear(50, 20),
            LeakyReLU(),
            Linear(20, n_tasks),
        )
        self.emb2label = Linear(emb_size, 1, bias=False)
        self.loss_form = BCELoss()
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def forward(self, x):
        c = self.x2c_model(x)
        c_logit = self.emb2label(c.permute(0, 2, 1).reshape(c.shape[0] * c.shape[2], -1))
        c_sem = c_logit.sigmoid()
        y = self.c2y_model(c)
        y_logit = self.emb2label(y.reshape(y.shape[0] * y.shape[2], -1))
        y_sem = y_logit.sigmoid()
        return c_sem, y_sem, c, y, c_logit, y_logit

    def training_step(self, batch, batch_no):
        loss_c, loss_y, loss, c_accuracy, y_accuracy = step(self, batch, self.loss_form)
        self.train_accuracy.append([c_accuracy, y_accuracy])
        self.train_loss.append([loss_c.item(), loss_y.item(), loss.item()])
        return loss

    def validation_step(self, batch, batch_no):
        loss_c, loss_y, loss, c_accuracy, y_accuracy = step(self, batch, self.loss_form)
        self.val_accuracy.append([c_accuracy, y_accuracy])
        self.val_loss.append([loss_c.item(), loss_y.item(), loss.item()])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.008)


class ToyNetEmbNorm(pl.LightningModule):
    def __init__(self, n_features, n_concepts, n_tasks, emb_size):
        super().__init__()
        self.save_hyperparameters()
        self.x2c_model = Sequential(
            Linear(n_features, 10),
            LeakyReLU(),
            Linear(10, 10),
            LeakyReLU(),
            ConceptEmbeddings(in_features=10, out_features=n_concepts, emb_size=emb_size, bias=True),
        )
        self.c2y_model = Sequential(
            Linear(n_concepts, 50),
            LeakyReLU(),
            Linear(50, 20),
            LeakyReLU(),
            Linear(20, n_tasks),
        )
        self.loss_form = BCELoss()
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def forward(self, x):
        c = self.x2c_model(x)
        c_emb = embedding_to_nesyemb(c)
        c_logit = semantics(c)
        c_sem = semantics(c_emb)
        y = self.c2y_model(c_emb.permute(0, 2, 1))
        y_logit = semantics(y)
        y_emb = embedding_to_nesyemb(y)
        y_sem = semantics(y_emb)
        return c_sem, y_sem, c, y, c_logit, y_logit

    def training_step(self, batch, batch_no):
        loss_c, loss_y, loss, c_accuracy, y_accuracy = step(self, batch, self.loss_form)
        self.train_accuracy.append([c_accuracy, y_accuracy])
        self.train_loss.append([loss_c.item(), loss_y.item(), loss.item()])
        return loss

    def validation_step(self, batch, batch_no):
        loss_c, loss_y, loss, c_accuracy, y_accuracy = step(self, batch, self.loss_form)
        self.val_accuracy.append([c_accuracy, y_accuracy])
        self.val_loss.append([loss_c.item(), loss_y.item(), loss.item()])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.008)


class ToyNetFuzzyExtra(pl.LightningModule):
    def __init__(self, n_features, n_concepts, n_tasks, emb_size):
        super().__init__()
        self.save_hyperparameters()
        self.x2c_model = Sequential(
            Linear(n_features, 10),
            LeakyReLU(),
            Linear(10, 10),
            LeakyReLU(),
            Linear(10, n_concepts * emb_size),
        )
        self.c2y_model = Sequential(
            Linear(n_concepts * emb_size, 50),
            LeakyReLU(),
            Linear(50, 20),
            LeakyReLU(),
            Linear(20, n_tasks),
            Sigmoid()
        )
        self.loss_form = BCELoss()
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def forward(self, x):
        c = self.x2c_model(x)
        c_logit = c[:, :self.hparams['n_concepts']]
        c_sem = torch.sigmoid(c_logit)
        c_free = torch.nn.functional.leaky_relu(c[:, self.hparams['n_concepts']:])
        c_out = torch.hstack([c_sem, c_free])
        y = self.c2y_model(c_out)
        y_logit = y
        return c_sem, y, c_out, y, c_logit, y_logit

    def training_step(self, batch, batch_no):
        loss_c, loss_y, loss, c_accuracy, y_accuracy = step(self, batch, self.loss_form)
        self.train_accuracy.append([c_accuracy, y_accuracy])
        self.train_loss.append([loss_c.item(), loss_y.item(), loss.item()])
        return loss

    def validation_step(self, batch, batch_no):
        loss_c, loss_y, loss, c_accuracy, y_accuracy = step(self, batch, self.loss_form)
        self.val_accuracy.append([c_accuracy, y_accuracy])
        self.val_loss.append([loss_c.item(), loss_y.item(), loss.item()])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.008)


class ToyNetFuzzy(pl.LightningModule):
    def __init__(self, n_features, n_concepts, n_tasks):
        super().__init__()
        self.save_hyperparameters()
        self.x2c_model = Sequential(
            Linear(n_features, 10),
            LeakyReLU(),
            Linear(10, 10),
            LeakyReLU(),
            Linear(10, n_concepts),
            Sigmoid()
        )
        self.c2y_model = Sequential(
            Linear(n_concepts, 60),
            LeakyReLU(),
            Linear(60, 20),
            LeakyReLU(),
            Linear(20, n_tasks),
            Sigmoid()
        )
        self.loss_form = BCELoss()
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def forward(self, x):
        c = self.x2c_model(x)
        y = self.c2y_model(c)
        return c, y, c, y, c, y

    def training_step(self, batch, batch_no):
        loss_c, loss_y, loss, c_accuracy, y_accuracy = step(self, batch, self.loss_form)
        self.train_accuracy.append([c_accuracy, y_accuracy])
        self.train_loss.append([loss_c.item(), loss_y.item(), loss.item()])
        return loss

    def validation_step(self, batch, batch_no):
        loss_c, loss_y, loss, c_accuracy, y_accuracy = step(self, batch, self.loss_form)
        self.val_accuracy.append([c_accuracy, y_accuracy])
        self.val_loss.append([loss_c.item(), loss_y.item(), loss.item()])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.008)


class ToyNetBool(pl.LightningModule):
    def __init__(self, n_features, n_concepts, n_tasks):
        super().__init__()
        self.save_hyperparameters()
        self.x2c_model = Sequential(
            Linear(n_features, 10),
            LeakyReLU(),
            Linear(10, 10),
            LeakyReLU(),
            Linear(10, n_concepts),
            Sigmoid()
        )
        self.c2y_model = Sequential(
            Linear(n_concepts, 60),
            LeakyReLU(),
            Linear(60, 20),
            LeakyReLU(),
            Linear(20, n_tasks),
            Sigmoid()
        )
        self.loss_form = BCELoss()
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def forward(self, x):
        c = self.x2c_model(x)
        y = self.c2y_model((c>0.5).float())
        return c, y, c, y, c, y

    def training_step(self, batch, batch_no):
        loss_c, loss_y, loss, c_accuracy, y_accuracy = step(self, batch, self.loss_form)
        self.train_accuracy.append([c_accuracy, y_accuracy])
        self.train_loss.append([loss_c.item(), loss_y.item(), loss.item()])
        return loss

    def validation_step(self, batch, batch_no):
        loss_c, loss_y, loss, c_accuracy, y_accuracy = step(self, batch, self.loss_form)
        self.val_accuracy.append([c_accuracy, y_accuracy])
        self.val_loss.append([loss_c.item(), loss_y.item(), loss.item()])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.008)


def step(model, batch, loss_form):
    x, c, y = batch
    c_probs, y_probs, _, _, _, _ = model(x)
    loss_c = loss_form(c_probs.ravel(), c.ravel())
    loss_y = loss_form(y_probs.ravel(), y.ravel())
    loss = loss_c + 0.5 * loss_y
    # loss = loss_y
    # compute accuracy
    c_accuracy, y_accuracy = compute_accuracy(c_probs, y_probs, c, y)
    print(f'Loss {loss:.4f}, c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}')
    return loss_c, loss_y, loss, c_accuracy, y_accuracy


def compute_accuracy(c_pred, y_pred, c_true, y_true):
    c_pred = c_pred.ravel().cpu().detach() > 0.5
    y_pred = y_pred.ravel().cpu().detach() > 0.5
    c_true = c_true.ravel().cpu().detach()
    y_true = y_true.ravel().cpu().detach()
    c_accuracy = accuracy_score(c_true, c_pred)
    y_accuracy = accuracy_score(y_true, y_pred)
    return c_accuracy, y_accuracy
