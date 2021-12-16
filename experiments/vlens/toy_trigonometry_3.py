import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from torch.nn import BCELoss, BCEWithLogitsLoss, Sequential, LeakyReLU, Linear
from torch.utils.data import DataLoader, TensorDataset
from torch_explain.nn import ConceptEmbeddings, context, semantics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch_explain.nn.vector_logic import NeSyLayer


def generate_data(size):
    # sample from normal distribution
    h = np.random.randn(size, 3) * 4 + 1
    w, y, z = h[:, 0], h[:, 1], h[:, 2]

    # raw features
    x = np.stack([
        np.sin(w) + w,
        np.cos(w) + w,
        np.sin(y) + y,
        np.cos(y) + y,
        np.sin(z) + z,
        np.cos(z) + z,
        w ** 2 + y ** 2 + z ** 2,
    ]).T

    # concetps
    c = np.stack([
        w > 0,
        y > 0,
        z > 0,
    ]).T

    # task(s)
    y = (w + y) > 0

    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)
    return x, c, y


def main():
    seed_everything(42)

    # parameters for data, model, and training
    batch_size = 3000
    batch_size_test = 1000
    max_epochs = 400

    # generate "trigonometric" train data set
    x, c, y = generate_data(batch_size)
    train_data = TensorDataset(x, c, y)
    train_dl = DataLoader(train_data, batch_size=batch_size)
    n_features, n_concepts, n_tasks = x.shape[1], c.shape[1], 1

    # generate "trigonometric" test data set
    x_test, c_test, y_test = generate_data(batch_size_test)

    # train model *without* embeddings (concepts are just scalars)
    # train model
    model_1 = TrigoNet(n_features, n_concepts, n_tasks)
    trainer = pl.Trainer(gpus=0, max_epochs=max_epochs)
    trainer.fit(model_1, train_dl)
    # freeze model and compute test accuracy
    model_1.freeze()
    c_logits, y_logits = model_1.forward(x_test)
    c_accuracy_1, y_accuracy_1 = compute_accuracy(torch.sigmoid(c_logits), torch.sigmoid(y_logits), c_test, y_test)
    print(f'c_acc: {c_accuracy_1:.4f}, y_acc: {y_accuracy_1:.4f}')

    # train model *with* embeddings (concepts are vectors)
    # train model
    model_2 = TrigoNetEmb(n_features, n_concepts, n_tasks)
    trainer = pl.Trainer(gpus=0, max_epochs=max_epochs)
    trainer.fit(model_2, train_dl)
    # freeze model and compute test accuracy
    model_2.freeze()
    c_sem, y_sem = model_2.forward(x_test)
    c_accuracy_2, y_accuracy_2 = compute_accuracy(c_sem, y_sem, c_test, y_test)
    print(f'c_acc: {c_accuracy_2:.4f}, y_acc: {y_accuracy_2:.4f}, ')

    # now have fun and plot a few results!
    # first, create dir to save plots
    result_dir = './results/trigonometry_3/'
    os.makedirs(result_dir, exist_ok=True)

    # count the number of trainable parameters for each model
    n_params_model_1 = sum(p.numel() for p in model_1.parameters())
    n_params_model_2 = sum(p.numel() for p in model_2.parameters())

    # get train accuracies (both for concepts and for tasks)
    c_train_loss_1, y_train_loss_1 = np.array(model_1.loss_list)[:, 0], np.array(model_1.loss_list)[:, 1]
    c_train_loss_2, y_train_loss_2 = np.array(model_2.loss_list)[:, 0], np.array(model_2.loss_list)[:, 1]

    # plot concept accuracy
    epochs = np.arange(len(model_1.loss_list))
    plt.figure()
    plt.title(f'Concept Accuracy\n'
              f'Test accuracy with "scalar" concepts={c_accuracy_1:.2f} (params: {n_params_model_1})\n'
              f'Test accuracy with "vector" concepts={c_accuracy_2:.2f} (params: {n_params_model_2})')
    sns.lineplot(epochs, c_train_loss_1, label='train accuracy "scalar"')
    sns.lineplot(epochs, c_train_loss_2, label='train accuracy "vector"')
    plt.ylim([0.6, 1.05])
    plt.xlabel('epochs')
    plt.ylabel('train accuracy (c)')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'c_accuracy.png'))
    plt.show()

    # plot task accuracy
    plt.figure()
    plt.title(f'Task Accuracy\n'
              f'Test accuracy with "scalar" concepts={y_accuracy_1:.2f} (params: {n_params_model_1})'
              f'\nTest accuracy with "vector" concepts={y_accuracy_2:.2f} (params: {n_params_model_2})')
    sns.lineplot(epochs, y_train_loss_1, label='train accuracy "scalar"')
    sns.lineplot(epochs, y_train_loss_2, label='train accuracy "vector"')
    plt.ylim([0.6, 1.05])
    plt.xlabel('epochs')
    plt.ylabel('train accuracy (y)')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'y_accuracy.png'))
    plt.show()

    # finally, inspect the "context" of concept embeddings
    c_emb_test = model_2.x2c_model(x_test)
    y_emb_test = model_2.c2y_model(c_emb_test)
    c_ctx = context(c_emb_test)
    y_ctx = context(y_emb_test).squeeze(1)
    # reduce dimensionality to get 2D plots
    c_ctxr = TSNE().fit_transform(c_ctx.reshape(3000, 5))
    y_ctxr = TSNE().fit_transform(y_ctx)
    c_ctxrr = c_ctxr.reshape(1000, 3, 2)

    # plot "context" for concepts
    plt.figure()
    plt.title(f'Concept "context" (test)')
    sns.scatterplot(c_ctxrr[:, 0, 0], c_ctxrr[:, 0, 1], label='concept #1')
    sns.scatterplot(c_ctxrr[:, 1, 0], c_ctxrr[:, 1, 1], label='concept #2')
    sns.scatterplot(c_ctxrr[:, 2, 0], c_ctxrr[:, 2, 1], label='concept #3')
    plt.xlabel('tsne dim #1')
    plt.ylabel('tsne dim #2')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'c_ctx.png'))
    plt.show()

    # plot "context" for tasks
    plt.figure()
    plt.title(f'Task "context" (test)')
    sns.scatterplot(y_ctxr[y_sem[:, 0]>=0.5, 0], y_ctxr[y_sem[:, 0]>=0.5, 1], label='class #1')
    sns.scatterplot(y_ctxr[y_sem[:, 0]<0.5, 0], y_ctxr[y_sem[:, 0]<0.5, 1], label='class #2')
    plt.xlabel('tsne dim #1')
    plt.ylabel('tsne dim #2')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'y_ctx.png'))
    plt.show()

    return


class TrigoNetEmb(pl.LightningModule):
    def __init__(self, n_features, n_concepts, n_tasks):
        super().__init__()
        self.x2c_model = Sequential(*[
            Linear(n_features, 10),
            LeakyReLU(),
            Linear(10, 10),
            LeakyReLU(),
            ConceptEmbeddings(in_features=10, out_features=n_concepts, emb_size=5, bias=True),
        ])
        self.c2y_model = NeSyLayer(5, n_concepts, 3, n_tasks)
        self.loss = BCELoss()
        self.loss_list = []

    def forward(self, x):
        c = self.x2c_model(x)
        y = self.c2y_model(c)
        c_sem = semantics(c)
        y_sem = semantics(y)
        return c_sem, y_sem

    def training_step(self, batch, batch_no):
        x, c, y = batch
        c_sem, y_sem = self(x)
        loss = self.loss(c_sem, c) + self.loss(y_sem.ravel(), y)
        c_accuracy, y_accuracy = compute_accuracy(c_sem, y_sem, c, y)
        print(f'Loss {loss:.4f}, c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}, ')
        self.loss_list.append([c_accuracy, y_accuracy])

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


class TrigoNet(pl.LightningModule):
    def __init__(self, n_features, n_concepts, n_tasks):
        super().__init__()
        self.x2c_model = Sequential(*[
            Linear(n_features, 10),
            LeakyReLU(),
            Linear(10, 10),
            LeakyReLU(),
            Linear(10, n_concepts),
        ])
        self.c2y_model = Sequential(*[
            Linear(n_concepts, 100),
            LeakyReLU(),
            Linear(100, 100),
            LeakyReLU(),
            Linear(100, n_tasks),
        ])
        self.loss = BCEWithLogitsLoss()
        self.loss_list = []

    def forward(self, x):
        c = self.x2c_model(x)
        y = self.c2y_model((c>0).float())
        return c, y

    def training_step(self, batch, batch_no):
        x, c, y = batch
        c_logits, y_logits = self(x)
        y_logits = y_logits.reshape(-1)
        loss = self.loss(c_logits, c) + self.loss(y_logits, y)

        # compute accuracy
        c_accuracy, y_accuracy = compute_accuracy(torch.sigmoid(c_logits), torch.sigmoid(y_logits), c, y)
        print(f'Loss {loss:.4f}, c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}, ')
        self.loss_list.append([c_accuracy, y_accuracy])

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def compute_accuracy(c_pred, y_pred, c_true, y_true):
    c_pred = c_pred.ravel().cpu().detach() > 0.5
    y_pred = y_pred.ravel().cpu().detach() > 0.5
    c_true = c_true.ravel().cpu().detach()
    y_true = y_true.ravel().cpu().detach()
    c_accuracy = accuracy_score(c_true, c_pred)
    y_accuracy = accuracy_score(y_true, y_pred)
    return c_accuracy, y_accuracy


if __name__ == '__main__':
    main()
