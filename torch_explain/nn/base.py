from abc import abstractmethod
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class BaseNet(pl.LightningModule):
    @abstractmethod
    def __init__(self, loss_criterion: _Loss, optimizer: Optimizer, lr: float = 1e-3):
        super().__init__()
        self.loss_criterion = loss_criterion
        self.optmizer = optimizer
        self.lr = lr
        self.register_buffer("trained", torch.tensor(False))
        self.need_pruning = False

    @abstractmethod
    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss_criterion(y_out, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss_criterion(y_out, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss_criterion(y_out, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @abstractmethod
    def configure_optimizers(self):
        return self.optmizer(self.parameters(), lr=self.lr)

    @abstractmethod
    def explain_class(self, x):
        pass
