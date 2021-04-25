import os
import glob
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.models import resnet18, ResNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from torchvision.models.resnet import BasicBlock

from torch_explain.logic.nn import explain_class
from torch_explain.nn import ConceptAwareness
from torch_explain.nn.functional import l1_loss


class Explainer(pl.LightningModule):
    def __init__(self, n_concepts: int, n_classes: int, base_net: nn.Module = resnet18(pretrained=False),
                 optimizer: str = 'adamw',
                 explainer_hidden: list = (10, 10), concept_loss: _Loss = nn.BCELoss(),
                 task_loss: _Loss = nn.NLLLoss(),
                 lr: float = 1e-2, concept_activation: str = 'sigmoid'):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.explainer_hidden = explainer_hidden
        self.concept_loss = concept_loss
        self.task_loss = task_loss
        self.optmizer = optimizer
        self.lr = lr
        self.base_net = base_net
        self.concept_activation = concept_activation
        self.need_pruning = False
        self.concepts = None

        # base net
        self.base_net = base_net
        if base_net is not None:
            self.base_net.fc = torch.nn.Linear(self.base_net.fc.in_features, n_concepts)
        else:
            self.base_net = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), torch.nn.MaxPool2d(2, 2, padding=1),
                nn.BatchNorm2d(8, momentum=0.7),
                nn.Conv2d(8, n_concepts, 3, padding=1), nn.ReLU(), torch.nn.MaxPool2d(2, 2, padding=1),
                nn.BatchNorm2d(n_concepts, momentum=0.7),
                nn.Flatten(),
                nn.Linear(10 * 57 * 57, 64), nn.LeakyReLU(),
                nn.Linear(64, n_concepts),
            )

        # explainer
        self.explainer_layers = []
        if len(explainer_hidden) > 0:
            self.explainer_layers.append(ConceptAwareness(n_concepts, explainer_hidden[0], activation=concept_activation))
            self.explainer_layers.append(nn.LeakyReLU())
            for i in range(len(explainer_hidden)):
                in_features = n_concepts if i == 0 else explainer_hidden[-1]
                self.explainer_layers.append(nn.Linear(in_features, explainer_hidden[i]))
                self.explainer_layers.append(nn.LeakyReLU())

            self.explainer_layers.append(nn.Linear(explainer_hidden[-1], n_classes))
            self.explainer_layers.append(ConceptAwareness(n_classes, n_classes, activation='identity', top=True))

        else:
            self.explainer_layers = [
                ConceptAwareness(n_concepts, n_classes, activation=concept_activation),
                ConceptAwareness(n_classes, n_classes, activation='identity', top=True),
            ]

        self.explainer = torch.nn.Sequential(*self.explainer_layers)

        self.save_hyperparameters()

    def forward(self, x):
        c_out = self.base_net(x)
        y_out = self.explainer(c_out)
        return torch.sigmoid(c_out), nn.functional.log_softmax(y_out, dim=1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        c_out, y_out = self.forward(x)
        # base
        y1h = nn.functional.one_hot(y.long()).to(torch.float)
        loss = self.concept_loss(c_out, y1h)
        accuracy = (y1h.eq(c_out > 0.5).sum(dim=1) == y1h.shape[1]).sum() / len(y)
        # explainer
        loss += 0.01 * self.task_loss(y_out, (y % 2 == 1).to(torch.long))
        accuracy2 = y_out.argmax(dim=1).eq(y % 2 == 1).sum() / len(y)
        loss += 0.0001 * l1_loss(self.explainer)
        # prune_logic_layers(self.explainer, self.trainer.current_epoch, 10, fan_in=5, device=self.device)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc2', accuracy2, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        c_out, y_out = self.forward(x)
        # base
        y1h = nn.functional.one_hot(y.long()).to(torch.float)
        loss = self.concept_loss(c_out, y1h)
        accuracy = (y1h.eq(c_out > 0.5).sum(dim=1) == y1h.shape[1]).sum() / len(y)
        # explainer
        loss += self.task_loss(y_out, (y % 2 == 1).to(torch.long))
        accuracy2 = y_out.argmax(dim=1).eq(y % 2 == 1).sum() / len(y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc2', accuracy2, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        c_out, y_out = self.forward(x)
        # base
        y1h = nn.functional.one_hot(y.long()).to(torch.float)
        loss = self.concept_loss(c_out, y1h)
        accuracy = (y1h.eq(c_out > 0.5).sum(dim=1) == y1h.shape[1]).sum() / len(y)
        # explainer
        loss += self.task_loss(y_out, (y % 2 == 1).to(torch.long))
        accuracy2 = y_out.argmax(dim=1).eq(y % 2 == 1).sum() / len(y)
        self.log('test_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_acc2', accuracy2, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optmizer == 'adamw':
            return [AdamW(self.base_net.parameters(), lr=1e-3), AdamW(self.explainer.parameters(), lr=1e-2)]

    def explain_class(self, x, y):
        c_out, y_out = self.forward(x.cuda())
        y2 = torch.zeros((y.shape[0], 2))
        y2[:, 0] = y.cpu() % 2 == 0
        y2[:, 1] = y.cpu() % 2 == 1
        class_explanation, class_explanations = explain_class(self.explainer.cpu(), (c_out.cpu()>0.5).to(torch.float), y2,
                                                              binary=False, target_class=0)
        print(class_explanation)
        return class_explanation


class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3), bias=False)


if __name__ == '__main__':
    seed_everything(42)
    # data
    size = 224
    resize = int(size * 0.9)
    mnist_transforms = transforms.Compose([
        transforms.Resize(size=resize),
        transforms.CenterCrop(size=size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = MNIST('../../data', train=True, download=True, transform=mnist_transforms)
    train_data, val_data, test_data = random_split(dataset, [50000, 5000, 5000])
    train_loader = DataLoader(train_data, batch_size=180)
    val_loader = DataLoader(val_data, batch_size=180)
    test_loader = DataLoader(test_data, batch_size=180)

    # model
    base_dir = f'{os.getcwd()}/MNIST/explainer'
    os.makedirs(base_dir, exist_ok=True)

    # training
    checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
    trainer = Trainer(max_epochs=15, gpus=1, auto_lr_find=True, deterministic=False,
                      check_val_every_n_epoch=1, default_root_dir=base_dir,
                      weights_save_path=base_dir, profiler="simple",
                      callbacks=[EarlyStopping(monitor='val_loss'), checkpoint_callback])

    path = glob.glob(f'{base_dir}/*.ckpt')
    # debug = True
    debug = False
    if path and not debug:
        model = Explainer.load_from_checkpoint(path[0])
    else:
        model = Explainer(n_concepts=10, n_classes=2, base_net=None)
        trainer.fit(model, val_loader, val_loader)

    model.freeze()
    trainer.test(model, test_dataloaders=test_loader)

    x, y = next(iter(test_loader))
    model.explain_class(x, y)
