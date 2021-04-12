import torch
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset


class BaseClassifier(pl.LightningModule):
    def __init__(self, n_classes: int, optimizer: str, loss: _Loss, lr: float, activation: callable, accuracy_score: callable):
        super().__init__()
        self.n_classes = n_classes
        self.loss = loss
        self.optmizer = optimizer
        self.lr = lr
        self.activation = activation
        self.accuracy_score = accuracy_score
        self.model = None
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return self.activation(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss(y_out, y)
        accuracy = self.accuracy_score(y_out, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss(y_out, y)
        accuracy = self.accuracy_score(y_out, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss(y_out, y)
        accuracy = self.accuracy_score(y_out, y)
        self.log('test_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optmizer == 'adamw':
            return AdamW(self.model.parameters(), lr=self.lr)

    def transform(self, dataloader: DataLoader, base_dir: str, extension: str):
        y_out_list, y_list = [], []
        for i_batch, (x, y) in enumerate(dataloader):
            y_out = self.model(x.to(self.device))
            y_out_list.append(y_out.cpu())
            y_list.append(y.cpu())
        y_out, y = torch.cat(y_out_list), torch.cat(y_list)
        if base_dir:
            dataset = TensorDataset(y_out, y)
            torch.save(dataset, f'{base_dir}/c2y_{extension}.pt')
            return dataset
        return y_out, y


def concept_accuracy(y_out, y):
    return (y.eq(y_out > 0.5).sum(dim=1) == y.shape[1]).sum() / len(y)


def task_accuracy(y_out, y):
    return y_out.argmax(dim=1).eq(y).sum() / len(y)
