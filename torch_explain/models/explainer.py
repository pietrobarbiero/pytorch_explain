from abc import abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW
from torchvision.datasets import MNIST

from torch_explain.logic import explain_class
from torch_explain.models.base import task_accuracy, BaseClassifier
from torch_explain.nn import Logic
from torch_explain.utils.pruning import l1_loss, prune_logic_layers


class BaseExplainer(BaseClassifier):
    def __init__(self, n_concepts: int, n_classes: int, optimizer: str = 'adamw', loss: _Loss = nn.NLLLoss(),
                 lr: float = 1e-2, activation: callable = F.log_softmax, accuracy_score: callable = task_accuracy,
                 explainer_hidden: list = (10, 10), prune_epoch: int = 10, fan_in: int = 5,
                 l1: float = 1e-5, concept_activation: str = 'sigmoid'):
        super().__init__(n_classes, optimizer, loss, lr, activation, accuracy_score)
        self.n_concepts = n_concepts
        self.concept_activation = concept_activation
        self.model_hidden = explainer_hidden
        self.prune_epoch = prune_epoch
        self.fan_in = fan_in
        self.l1 = l1
        self.model = None
        self.save_hyperparameters()

    def forward(self, x):
        y_out = self.model(x)
        return self.activation(y_out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss(y_out, y) + self.l1 * l1_loss(self.model)
        accuracy = self.accuracy_score(y_out, y)
        prune_logic_layers(self.model, self.trainer.current_epoch, self.prune_epoch,
                           fan_in=self.fan_in, device=self.device)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss(y_out, y) + self.l1 * l1_loss(self.model)
        accuracy = self.accuracy_score(y_out, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss(y_out, y) + self.l1 * l1_loss(self.model)
        accuracy = self.accuracy_score(y_out, y)
        self.log('test_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optmizer == 'adamw':
            return AdamW(self.model.parameters(), lr=self.lr)

    @abstractmethod
    def explain_class(self, x, y, target_class, topk_explanations=3):
        pass


class MuExplainer(BaseExplainer):
    def __init__(self, n_concepts: int, n_classes: int, optimizer: str = 'adamw', loss: _Loss = nn.NLLLoss(),
                 lr: float = 1e-2, activation: callable = F.log_softmax, accuracy_score: callable = task_accuracy,
                 explainer_hidden: list = (10, 10), prune_epoch: int = 10, fan_in: int = 5, l1: float = 1e-5,
                 concept_activation: str = 'sigmoid'):
        super().__init__(n_concepts, n_classes, optimizer, loss, lr, activation,
                         accuracy_score, explainer_hidden, prune_epoch, fan_in, l1, concept_activation)

        self.model_layers = []
        if len(explainer_hidden) > 0:
            self.model_layers.append(Logic(n_concepts, explainer_hidden[0], activation=concept_activation))
            self.model_layers.append(nn.LeakyReLU())
            for i in range(len(explainer_hidden)):
                in_features = n_concepts if i == 0 else explainer_hidden[-1]
                self.model_layers.append(nn.Linear(in_features, explainer_hidden[i]))
                self.model_layers.append(nn.LeakyReLU())

            self.model_layers.append(nn.Linear(explainer_hidden[-1], n_classes))
            self.model_layers.append(Logic(n_classes, n_classes, activation='identity', top=True))

        else:
            self.model_layers = [
                Logic(n_concepts, n_classes, activation=concept_activation),
                Logic(n_classes, n_classes, activation='identity', top=True),
            ]

        self.model = torch.nn.Sequential(*self.model_layers)

        self.save_hyperparameters()

    def explain_class(self, x, y, target_class, topk_explanations=3):
        return explain_class(self.model.cpu(), x, y, binary=False,
                             target_class=target_class, topk_explanations=topk_explanations)


class MNIST_C_to_Y(MNIST):

    def __getitem__(self, index: int):
        target = int(self.targets[index])
        target1h = torch.zeros(10)
        target1h[target] = 1
        return target1h, torch.tensor(target % 2 == 1, dtype=torch.long)
