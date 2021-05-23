import torch
from PIL import Image
from torch import nn
from torch.nn.modules.loss import _Loss
from torchvision.datasets import MNIST
from torchvision.models import resnet18, ResNet
from torchvision.models.resnet import BasicBlock

from torch_explain.models.base import BaseClassifier, concept_accuracy


class BlackBoxSimple(BaseClassifier):
    def __init__(self, n_concepts: int, optimizer: str = 'adamw', concept_loss: _Loss = nn.BCEWithLogitsLoss(),
                 lr: float = 1e-3, activation: callable = torch.sigmoid, accuracy_score: callable = concept_accuracy):
        super().__init__(n_concepts, optimizer, concept_loss, lr, activation, accuracy_score)

        # base net
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), torch.nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(8, momentum=0.7),
            nn.Conv2d(8, n_concepts, 3, padding=1), nn.ReLU(), torch.nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(n_concepts, momentum=0.7),
            nn.Flatten(),
            nn.Linear(10 * 57 * 57, 64), nn.LeakyReLU(),
            nn.Linear(64, n_concepts),
        )
        self.save_hyperparameters()


class BlackBoxResNet18(BaseClassifier):
    def __init__(self, n_concepts: int, model: nn.Module = resnet18(pretrained=False),
                 optimizer: str = 'adamw', concept_loss: _Loss = nn.BCEWithLogitsLoss(),
                 lr: float = 1e-3, activation: callable = torch.sigmoid, accuracy_score: callable = concept_accuracy):
        super().__init__(n_concepts, optimizer, concept_loss, lr, activation, accuracy_score)
        # base net
        self.model = model
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, n_concepts)
        self.save_hyperparameters()


class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3), bias=False)


class MNIST_X_to_C(MNIST):

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = int(self.targets[index])
        target1h = torch.zeros(10)
        target1h[target] = 1

        return img, target1h
