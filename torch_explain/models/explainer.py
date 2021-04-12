from abc import abstractmethod

import torch
from sklearn.metrics import accuracy_score
from torch import nn
import torch.nn.functional as F
from torch.nn import Dropout
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from torch_explain.logic import explain_class, test_explanation, complexity
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
    def explain_class(self, x, y, target_class, topk_explanations=3, **kwargs):
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
            for i in range(1, len(explainer_hidden)):
                in_features = explainer_hidden[i - 1]
                self.model_layers.append(Dropout())
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

    def transform(self, dataloader: DataLoader, x_to_bool: bool = True):
        x_list, y_out_list, y_list = [], [], []
        for i_batch, (x, y) in enumerate(dataloader):
            y_out = self.model(x.to(self.device))
            x_list.append(x.cpu())
            y_out_list.append(y_out.cpu())
            y_list.append(y.cpu())
        x, y_out, y = torch.cat(x_list), torch.cat(y_out_list), torch.cat(y_list)

        if x_to_bool:
            x = (x.cpu() > 0.5).to(torch.float)
        y1h = F.one_hot(y)
        return x, y_out, y1h

    def explain_class(self, val_dataloaders: DataLoader, test_dataloaders: DataLoader,
                      target_class, concept_names=None, topk_explanations=3,
                      max_accuracy: bool = False, x_to_bool: bool = True, y_to_one_hot: bool = True):

        x_val, y_val_out, y_val_1h = self.transform(val_dataloaders)
        x_test, y_test_out, y_test_1h = self.transform(test_dataloaders)

        class_explanation, explanations = explain_class(self.model.cpu(), x_val, y_val_1h,
                                                        target_class=target_class,
                                                        topk_explanations=topk_explanations,
                                                        concept_names=concept_names,
                                                        max_accuracy=max_accuracy)
        accuracy, y_formula = test_explanation(class_explanation, target_class=1, x=x_test, y=y_test_1h,
                                               concept_names=concept_names)
        explanation_fidelity = accuracy_score(y_test_1h.argmax(dim=1), y_formula)
        explanation_complexity = complexity(class_explanation)
        results = {
            'target_class': target_class,
            'explanation': class_explanation,
            'explanation_accuracy': accuracy,
            'explanation_fidelity': explanation_fidelity,
            'explanation_complexity': explanation_complexity,
        }
        return results


class MNIST_C_to_Y(MNIST):

    def __getitem__(self, index: int):
        target = int(self.targets[index])
        target1h = torch.zeros(10)
        target1h[target] = 1
        return target1h, torch.tensor(target % 2 == 1, dtype=torch.long)
