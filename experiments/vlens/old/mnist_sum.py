import os
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.nn import Conv2d, BCELoss, Sequential, LeakyReLU, Linear
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch_explain as te
from experiments.data.load_datasets import load_vector_mnist
from torch_explain.nn import ConceptEmbeddings, context, semantics, to_boolean
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


def main():
    seed_everything(42)
    batch_size = 512

    train_data, test_data, concept_names, label_names = load_vector_mnist('../data')
    train_dl = DataLoader(train_data, batch_size=batch_size)
    test_dl = DataLoader(test_data, batch_size=batch_size)

    model = ResNetMNISTsum()
    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(model, train_dl)

    result_dir = './results/mnist_sum/'
    os.makedirs(result_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(result_dir, 'model.pt'))

    return


class ResNetMNISTsum(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = ConceptEmbeddings(in_features=512, out_features=10, emb_size=10, bias=True)
        self.c2y_model = Sequential(*[
            Linear(20, 20),
            LeakyReLU(),
            Linear(20, 19),
        ])
        self.loss = BCELoss()

    def forward(self, x1, x2):
        c1, c2, c = self.forward_cnn(x1, x2)
        y = self.forward_mlp(c)
        return c1, c2, c, y

    def forward_cnn(self, x1, x2):
        c1 = self.model(x1)
        c2 = self.model(x2)
        c = torch.concat([c1, c2], dim=1)
        return c1, c2, c

    def forward_mlp(self, c):
        c = to_boolean(c, true_norm=1, false_norm=2)
        return self.c2y_model(c.permute(0, 2, 1)).permute(0, 2, 1)

    def training_step(self, batch, batch_no):
        x1, x2, c1, c2, y = batch
        c1_emb, c2_emb, c_emb, y_emb = self(x1, x2)
        c1_logits = semantics(c1_emb)
        c2_logits = semantics(c2_emb)
        y_logits = semantics(y_emb)
        loss = self.loss(c1_logits, c1) + self.loss(c2_logits, c2) + self.loss(y_logits, y)

        # compute accuracy
        c_pred = torch.concat([c1_logits.reshape(-1), c2_logits.reshape(-1)]).cpu().detach() > 0.5
        y_pred = y_logits.reshape(-1).cpu().detach() > 0.5
        c_true = torch.concat([c1.reshape(-1), c2.reshape(-1)]).cpu().detach()
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
