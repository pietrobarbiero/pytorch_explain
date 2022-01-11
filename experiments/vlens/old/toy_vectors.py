import joblib
import os
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, normalized_mutual_info_score
from torch.nn import BCELoss, Sequential, LeakyReLU, Linear, Sigmoid
from torch.utils.data import DataLoader, TensorDataset
from torch_explain.nn import ConceptEmbeddings, semantics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch_explain.nn.vector_logic import NeSyLayer, to_boolean, embedding_to_nesyemb


def generate_data(size):
    # sample from normal distribution
    emb_size = 2
    v1 = np.random.randn(size, emb_size) * 2
    v2 = np.ones(emb_size)
    v3 = np.random.randn(size, emb_size) * 2
    v4 = -np.ones(emb_size)
    x = np.hstack([v1+v3, v1-v3])
    c = np.stack([
        np.dot(v1, v2).ravel() > 0,
        np.dot(v3, v4).ravel() > 0,
    ]).T
    y = (v1*v3).sum(axis=-1) > 0

    # clf = RandomForestClassifier(random_state=42)
    # cross_val_score(clf, x, c)

    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)
    return x, c, y


def main():
    # parameters for data, model, and training
    batch_size = 3000
    batch_size_test = 1000
    max_epochs = 3000
    gpu = 1
    cv = 5
    result_dir = './results/toy_vectors_hard_3/'
    os.makedirs(result_dir, exist_ok=True)

    results = {}
    for split in range(cv):
        print(f'Experiment {split+1}/{cv}')

        seed_everything(split)

        # generate "trigonometric" train data set
        x, c, y = generate_data(batch_size)
        train_data = TensorDataset(x, c, y)
        train_dl = DataLoader(train_data, batch_size=batch_size)
        n_features, n_concepts, n_tasks = x.shape[1], c.shape[1], 1

        # generate "trigonometric" test data set
        x_test, c_test, y_test = generate_data(batch_size_test)

        # train model *with* embeddings (concepts are vectors)
        # train model
        model_2 = TrigoNetEmb(n_features, n_concepts, n_tasks)
        trainer = pl.Trainer(gpus=gpu, max_epochs=max_epochs)
        trainer.fit(model_2, train_dl)
        # freeze model and compute test accuracy
        model_2.freeze()
        c_sem, y_sem, _ = model_2.forward(x_test)
        c_accuracy_2, y_accuracy_2 = compute_accuracy(c_sem, y_sem, c_test, y_test)
        print(f'c_acc: {c_accuracy_2:.4f}, y_acc: {y_accuracy_2:.4f}')

        # train model *without* embeddings (concepts are just *fuzzy* scalars)
        # train model
        model_3 = TrigoNet(n_features, n_concepts, n_tasks, bool=False)
        trainer = pl.Trainer(gpus=gpu, max_epochs=max_epochs)
        trainer.fit(model_3, train_dl)
        # freeze model and compute test accuracy
        model_3.freeze()
        c_logits, y_logits = model_3.forward(x_test)
        c_accuracy_3, y_accuracy_3 = compute_accuracy(c_logits, y_logits, c_test, y_test)
        print(f'c_acc: {c_accuracy_3:.4f}, y_acc: {y_accuracy_3:.4f}')

        # train model *without* embeddings (concepts are just *Boolean* scalars)
        # train model
        model_1 = TrigoNet(n_features, n_concepts, n_tasks, bool=True)
        trainer = pl.Trainer(gpus=gpu, max_epochs=max_epochs)
        trainer.fit(model_1, train_dl)
        # freeze model and compute test accuracy
        model_1.freeze()
        c_logits_1, y_logits_1 = model_1.forward(x_test)
        c_accuracy_1, y_accuracy_1 = compute_accuracy(c_logits_1, y_logits_1, c_test, y_test)
        print(f'c_acc: {c_accuracy_1:.4f}, y_acc: {y_accuracy_1:.4f}')

        # model bool
        c1_train = model_1.x2c_model(x)
        c1_test = model_1.x2c_model(x_test)
        y1_train = model_1.c2y_model(c1_train)
        y1_test = model_1.c2y_model(c1_test)
        # model fuzzy
        c3_train = model_3.x2c_model(x)
        c3_test = model_3.x2c_model(x_test)
        y3_train = model_3.c2y_model(c3_train)
        y3_test = model_3.c2y_model(c3_test)
        # model embeddings
        c2_train = model_2.x2c_model(x)
        c2_test = model_2.x2c_model(x_test)
        y2_train = model_2.c2y_model(to_boolean(c2_train, 2, 1).permute(0, 2, 1)).permute(0, 2, 1)
        y2_test = model_2.c2y_model(to_boolean(c2_test, 2, 1).permute(0, 2, 1)).permute(0, 2, 1)

        results[f'{split}'] = {
            'x_train': x,
            'c_train': c,
            'y_train': y,
            'x_test': x_test,
            'c_test': c_test,
            'y_test': y_test,
            'c_bool_train': c1_train,
            'c_bool_test': c1_test,
            'y_bool_train': y1_train,
            'y_bool_test': y1_test,
            'c_fuzzy_train': c3_train,
            'c_fuzzy_test': c3_test,
            'y_fuzzy_train': y3_train,
            'y_fuzzy_test': y3_test,
            'c_emb_train': c2_train,
            'c_emb_test': c2_test,
            'y_emb_train': y2_train,
            'y_emb_test': y2_test,
            'train_acc_bool': model_1.loss_list,
            'train_acc_fuzzy': model_3.loss_list,
            'train_acc_emb': model_2.loss_list,
            'mi_xc_bool': model_1.mi_xc,
            'mi_cy_bool': model_1.mi_cy,
            'mi_xc_fuzzy': model_3.mi_xc,
            'mi_cy_fuzzy': model_3.mi_cy,
            'mi_xc_emb': model_2.mi_xc,
            'mi_cy_emb': model_2.mi_cy,
            'trainable_params_bool': sum(p.numel() for p in model_1.parameters()),
            'trainable_params_fuzzy': sum(p.numel() for p in model_1.parameters()),
            'trainable_params_emb': sum(p.numel() for p in model_2.parameters()),
        }

        # save results
        joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

    return


class TrigoNetEmb(pl.LightningModule):
    def __init__(self, n_features, n_concepts, n_tasks):
        super().__init__()
        self.x2c_model = Sequential(*[
            Linear(n_features, 10),
            LeakyReLU(),
            Linear(10, 10),
            LeakyReLU(),
            ConceptEmbeddings(in_features=10, out_features=n_concepts, emb_size=2, bias=True),
        ])
        # self.c2y_model = NeSyLayer(5, n_concepts, n_concepts, n_tasks)
        self.c2y_model = Sequential(*[
            Linear(n_concepts, 50),
            LeakyReLU(),
            Linear(50, 20),
            LeakyReLU(),
            Linear(20, n_tasks),
        ])
        self.loss = BCELoss()
        self.loss_list = []
        self.mi_xc = []
        self.mi_cy = []

    def forward(self, x):
        c = self.x2c_model(x)
        # y = self.c2y_model(to_boolean(c, 2, 1))
        # y = self.c2y_model(to_boolean(c, 2, 1).permute(0, 2, 1)).permute(0, 2, 1)
        y = self.c2y_model(embedding_to_nesyemb(c).permute(0, 2, 1)).permute(0, 2, 1)
        c_sem = semantics(embedding_to_nesyemb(c))
        # c_sem = torch.norm(c, dim=-1) > 1.5
        # y_sem = semantics(y)
        # y_sem = torch.exp(-torch.norm(y, p=2, dim=-1))
        y_sem = semantics(embedding_to_nesyemb(y))
        return c_sem, y_sem, to_boolean(c, 2, 1)

    def training_step(self, batch, batch_no):
        x, c, y = batch
        c_sem, y_sem, c_emb = self(x)
        loss = self.loss(c_sem, c) + 0.5*self.loss(y_sem.ravel(), y)
        c_accuracy, y_accuracy = compute_accuracy(c_sem, y_sem, c, y)
        c_emb = c_emb.cpu().detach().reshape(c.shape[0], -1)
        mi_xc = np.mean([mutual_info_regression(x.cpu(), c_emb[:, i]).mean() for i in range(c_emb.shape[1])])
        mi_cy = mutual_info_classif(c_emb, y.cpu()).mean()
        self.mi_xc.append(mi_xc)
        self.mi_cy.append(mi_cy)
        print(f'Loss {loss:.4f}, c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}, mi_xc: {mi_xc:.4f}, mi_cy: {mi_cy:.4f}')
        self.loss_list.append([c_accuracy, y_accuracy])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.008)


class TrigoNet(pl.LightningModule):
    def __init__(self, n_features, n_concepts, n_tasks, bool):
        super().__init__()
        self.x2c_model = Sequential(*[
            Linear(n_features, 10),
            LeakyReLU(),
            Linear(10, 10),
            LeakyReLU(),
            Linear(10, n_concepts),
            Sigmoid()
        ])
        self.c2y_model = Sequential(*[
            Linear(n_concepts, 60),
            LeakyReLU(),
            Linear(60, 20),
            LeakyReLU(),
            Linear(20, n_tasks),
            Sigmoid()
        ])
        self.loss = BCELoss()
        self.bool = bool
        self.loss_list = []
        self.mi_xc = []
        self.mi_cy = []

    def forward(self, x):
        c = self.x2c_model(x)
        if self.bool:
            y = self.c2y_model((c>0.5).float())
        else:
            y = self.c2y_model(c)
        return c, y

    def training_step(self, batch, batch_no):
        x, c, y = batch
        c_logits, y_logits = self(x)
        y_logits = y_logits.reshape(-1)
        loss = self.loss(c_logits, c) + 0.5*self.loss(y_logits, y)
        # compute accuracy
        c2 = c.cpu()
        if self.bool:
            c2 = c2>0.5
        mi_xc = np.mean([mutual_info_regression(x.cpu(), c2[:, i]).mean() for i in range(c2.shape[1])])
        mi_cy = mutual_info_classif(c2, y.cpu()).mean()
        self.mi_xc.append(mi_xc)
        self.mi_cy.append(mi_cy)
        c_accuracy, y_accuracy = compute_accuracy(c_logits, y_logits, c, y)
        print(f'Loss {loss:.4f}, c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}, mi_xc: {mi_xc:.4f}, mi_cy: {mi_cy:.4f}')
        self.loss_list.append([c_accuracy, y_accuracy])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.008)


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
