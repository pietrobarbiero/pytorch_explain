import os
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.nn import Conv2d, BCELoss, Sequential, LeakyReLU, Linear
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch_explain as te
from experiments.data.load_datasets import load_dsprites
from torch_explain.nn import ConceptEmbeddings, context, semantics, to_boolean
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


def main():
    seed_everything(42)
    batch_size = 256

    train_data, val_data, test_data, concept_names = load_dsprites('../data')

    train_dl = DataLoader(train_data, batch_size=batch_size)
    # test_dl = DataLoader(test_data, batch_size=batch_size)

    model = ResNetdSprites()
    # c_pred, y_pred = model(train_data.tensors[0][:100])
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(model, train_dl)

    result_dir = './results/dsprites_2d/'
    os.makedirs(result_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(result_dir, 'model.pt'))

    return


class ResNetdSprites(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = ConceptEmbeddings(in_features=512, out_features=9, emb_size=2, bias=True)
        self.c2y_model = Sequential(*[
            Linear(9, 20),
            LeakyReLU(),
            Linear(20, 3),
        ])
        self.loss = BCELoss()

    def forward(self, x):
        c = self.model(x)
        y = self.forward_mlp(c)
        return c, y

    def forward_mlp(self, c):
        c = to_boolean(c, true_norm=1, false_norm=2)
        return self.c2y_model(c.permute(0, 2, 1)).permute(0, 2, 1)

    def training_step(self, batch, batch_no):
        x, c, y = batch
        c_emb, y_emb = self(x)
        c_logits = semantics(c_emb)
        y_logits = semantics(y_emb)
        loss = self.loss(c_logits, c) + self.loss(y_logits, y)

        # compute accuracy
        c_pred = c_logits.reshape(-1).cpu().detach() > 0.5
        y_pred = y_logits.reshape(-1).reshape(-1).cpu().detach() > 0.5
        c_true = c.reshape(-1).cpu().detach()
        y_true = y.reshape(-1).cpu().detach()
        c_accuracy = accuracy_score(c_true, c_pred)
        y_accuracy = accuracy_score(y_true, y_pred)
        c_f1 = f1_score(c_true>0.5, c_pred)
        y_f1 = f1_score(y_true, y_pred)
        print(f'Loss {loss:.4f}, '
              f'c_f1: {c_f1:.4f}, y_f1: {y_f1:.4f}, '
              f'c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}, ')

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.005)


if __name__ == '__main__':
    main()
