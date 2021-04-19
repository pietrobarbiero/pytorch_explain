from abc import abstractmethod
from typing import List

import torch
from sklearn.metrics import accuracy_score
from torch import nn
import torch.nn.functional as F
from torch.nn import Dropout
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import seaborn as sns

from torch_explain.logic.nn import explain_class
from torch_explain.logic.metrics import  test_explanation, complexity
from torch_explain.models.base import task_accuracy, BaseClassifier, concept_accuracy
from torch_explain.nn import Conceptizator
from torch_explain.nn.functional import l1_loss
from torch_explain.nn.logic import LogicAttention


class BaseExplainer(BaseClassifier):
    def __init__(self, n_concepts: int, n_classes: int, optimizer: str = 'adamw', loss: _Loss = nn.BCEWithLogitsLoss(),
                 lr: float = 1e-2, activation: callable = F.log_softmax, accuracy_score: callable = concept_accuracy,
                 explainer_hidden: list = (10, 10), l1: float = 1e-5):
        super().__init__(n_classes, optimizer, loss, lr, activation, accuracy_score)
        self.n_concepts = n_concepts
        self.model_hidden = explainer_hidden
        self.l1 = l1
        self.model = None
        self.save_hyperparameters()

    def forward(self, x):
        y_out = self.model(x)
        return y_out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss(y_out, y) + self.l1 * l1_loss(self.model)
        accuracy = self.accuracy_score(y_out, y)
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
        accuracy = self.accuracy_score(y_out, y)
        self.log('test_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        if self.optmizer == 'adamw':
            return AdamW(self.model.parameters(), lr=self.lr)

    @abstractmethod
    def explain_class(self, x, y, target_class, topk_explanations=3, **kwargs):
        pass


class MuExplainer(BaseExplainer):
    def __init__(self, n_concepts: int, n_classes: int, optimizer: str = 'adamw', loss: _Loss = nn.BCEWithLogitsLoss(),
                 lr: float = 1e-2, activation: callable = F.log_softmax, accuracy_score: callable = task_accuracy,
                 explainer_hidden: list = (8, 3), l1: float = 1e-5):
        super().__init__(n_concepts, n_classes, optimizer, loss, lr, activation,
                         accuracy_score, explainer_hidden, l1)

        self.model_layers = []
        self.model_layers.append(LogicAttention(n_concepts, explainer_hidden[0], n_classes, n_heads=1))
        self.model_layers.append(torch.nn.LeakyReLU())
        self.model_layers.append(Dropout())
        for i in range(1, len(explainer_hidden)):
            self.model_layers.append(LogicAttention(explainer_hidden[i - 1], explainer_hidden[i], n_classes))
            self.model_layers.append(torch.nn.LeakyReLU())
            self.model_layers.append(Dropout())

        self.model_layers.append(LogicAttention(explainer_hidden[-1], 1, n_classes, top=True))
        # self.model_layers.append(torch.nn.LogSoftmax(dim=1))
        # self.model_layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*self.model_layers)

        self.save_hyperparameters()

    def transform(self, dataloader: DataLoader, x_to_bool: bool = True, y_to_one_hot: bool = True):
        x_list, y_out_list, y_list = [], [], []
        for i_batch, (x, y) in enumerate(dataloader):
            y_out = self.forward(x.to(self.device))
            x_list.append(x.cpu())
            y_out_list.append(y_out.cpu())
            y_list.append(y.cpu())
        x, y_out, y = torch.cat(x_list), torch.cat(y_out_list), torch.cat(y_list)

        if x_to_bool:
            x = (x.cpu() > 0.5).to(torch.float)
        # if y_to_one_hot:
        #     y = F.one_hot(y)
        return x, y_out, y

    def explain_class(self, val_dataloaders: DataLoader, test_dataloaders: DataLoader,
                      target_class: int, concept_names: List = None,
                      topk_explanations: int = 3, max_minterm_complexity: int = None,
                      max_accuracy: bool = False, x_to_bool: bool = True, y_to_one_hot: bool = False):

        x_val, y_val_out, y_val_1h = self.transform(val_dataloaders)
        x_test, y_test_out, y_test_1h = self.transform(test_dataloaders)

        class_explanation, explanations = explain_class(self.model.cpu(), x_val, y_val_1h,
                                                        target_class=target_class,
                                                        topk_explanations=topk_explanations,
                                                        max_minterm_complexity=max_minterm_complexity,
                                                        concept_names=concept_names,
                                                        max_accuracy=max_accuracy)
        accuracy, y_formula = test_explanation(class_explanation, target_class=target_class,
                                               x=x_test, y=y_test_1h[:, target_class],
                                               concept_names=concept_names)
        explanation_fidelity = accuracy_score(y_test_out.argmax(dim=1).eq(target_class), y_formula)
        explanation_complexity = complexity(class_explanation)
        results = {
            'target_class': target_class,
            'explanation': class_explanation,
            'explanation_accuracy': accuracy,
            'explanation_fidelity': explanation_fidelity,
            'explanation_complexity': explanation_complexity,
        }
        return results

    # def inspect(self, dataloader):
    #     x, y_out, y_1h = self.transform(dataloader, y_to_one_hot=False)
    #     h_prev = x
    #     n_layers = len([1 for module in self.model.modules() if isinstance(module, LogicAttention)])-1
    #     layer_id = 1
    #     plt.figure(figsize=[10, 4])
    #     for module in self.model.modules():
    #         if isinstance(module, nn.Sequential) or isinstance(module, Conceptizator):
    #             continue
    #         h = module(h_prev)
    #         if isinstance(module, LogicAttention) and not module.top:
    #             plt.subplot(1, n_layers, layer_id)
    #             plt.title(f'Layer {layer_id}')
    #             sns.scatterplot(x=h_prev.view(-1), y=module.conceptizator.concepts.view(-1))
    #             layer_id += 1
    #         h_prev = h
    #     plt.tight_layout()
    #     plt.show()
    #     return


class MNIST_C_to_Y(MNIST):

    def __getitem__(self, index: int):
        target = int(self.targets[index])
        target1h = torch.zeros(10)
        target1h[target] = 1
        return target1h, torch.tensor(target % 2 == 1, dtype=torch.long)
