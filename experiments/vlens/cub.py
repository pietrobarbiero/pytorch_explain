import joblib
import os
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.nn import BCELoss, CrossEntropyLoss, Sequential, LeakyReLU, Linear, Sigmoid, Softmax
from torch.nn.functional import one_hot
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset
from torch_explain.nn import ConceptEmbeddings, semantics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch_explain.nn.vector_logic import NeSyLayer, to_boolean
from experiments.data.load_datasets import load_cub_full


def map_to_emb():
    batch_size = 512
    num_workers = 10
    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    train_data, val_data, test_data, concept_names, class_names = load_cub_full('../data/CUB200')
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    preds


def get_preds(trainer, model, test_dl):
    preds = trainer.predict(model, test_dl)
    c_logits, y_logits, c_emb = [], [], []
    for pred in preds:
        c_logits.append(pred[0])
        y_logits.append(pred[1])
        if len(pred) == 3:
            c_emb.append(pred[2])
    c_preds = torch.vstack(c_logits)
    y_preds = torch.vstack(y_logits)
    if len(pred) == 3:
        c_emb = torch.vstack(c_emb)
    return c_preds, y_preds, c_emb


def main():
    batch_size = 512
    num_workers = 10
    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    data, concept_names, class_names = load_cub_full('../data/CUB200')
    train_data = TensorDataset(data['x_train'], data['c_train'], data['y_train'])
    val_data = TensorDataset(data['x_val'], data['c_val'], data['y_val'])
    test_data = TensorDataset(data['x_test'], data['c_test'], data['y_test'])
    train_dl = DataLoader(train_data, batch_size=batch_size)
    val_dl = DataLoader(val_data, batch_size=batch_size)
    test_dl = DataLoader(test_data, batch_size=batch_size)
    n_features, n_concepts, n_tasks = data['x_train'].shape[1], len(concept_names), len(class_names)
    x_test, c_test, y_test = test_data.tensors

    # parameters for data, model, and training
    batch_size = 512
    max_epochs = 30
    gpu = 1
    cv = 5
    num_workers = 10
    result_dir = './results/cub/'
    os.makedirs(result_dir, exist_ok=True)

    results = {}
    for split in range(cv):
        print(f'Experiment {split+1}/{cv}')

        seed_everything(split)

        # train model *without* embeddings (concepts are just *fuzzy* scalars)
        # train model
        model_3 = TrigoNet(n_features, n_concepts, n_tasks, bool=False)
        trainer = pl.Trainer(gpus=gpu, max_epochs=max_epochs)
        trainer.fit(model_3, train_dl, val_dl)
        # freeze model and compute test accuracy
        model_3.freeze()
        c_logits, y_logits = model_3.forward(test_data.tensors)
        c_accuracy_3, y_accuracy_3 = compute_accuracy(c_logits, y_logits, c_test, y_test)
        print(f'c_acc: {c_accuracy_3:.4f}, y_acc: {y_accuracy_3:.4f}')

        # train model *with* embeddings (concepts are vectors)
        # train model
        model_2 = TrigoNetEmb(n_features, n_concepts, n_tasks)
        trainer = pl.Trainer(gpus=gpu, max_epochs=max_epochs)
        trainer.fit(model_2, train_dl, val_dl)
        # freeze model and compute test accuracy
        model_2.freeze()
        c_sem, y_sem, _ = model_2.forward(test_data.tensors)
        c_accuracy_2, y_accuracy_2 = compute_accuracy(c_sem, y_sem, c_test, y_test)
        print(f'c_acc: {c_accuracy_2:.4f}, y_acc: {y_accuracy_2:.4f}')

        # train model *without* embeddings (concepts are just *Boolean* scalars)
        # train model
        model_1 = TrigoNet(n_features, n_concepts, n_tasks, bool=True)
        trainer = pl.Trainer(gpus=gpu, max_epochs=max_epochs)
        trainer.fit(model_1, train_dl, val_dl)
        # freeze model and compute test accuracy
        model_1.freeze()
        c_logits_1, y_logits_1 = model_1.forward(test_data.tensors)
        c_accuracy_1, y_accuracy_1 = compute_accuracy(c_logits_1, y_logits_1, c_test, y_test)
        print(f'c_acc: {c_accuracy_1:.4f}, y_acc: {y_accuracy_1:.4f}')

        # model bool
        c1_train, y1_train = model_3.forward(train_data.tensors)
        c1_test, y1_test = model_3.forward(train_data.tensors)
        # model fuzzy
        c3_train, y3_train, _ = get_preds(trainer, model_3, train_dl)
        c3_test, y3_test, _ = get_preds(trainer, model_3, test_dl)
        # model embeddings
        c2_train, y2_train, c_emb_train = get_preds(trainer, model_2, train_dl)
        c2_test, y2_test, c_emb_test = get_preds(trainer, model_2, test_dl)

        results[f'{split}'] = {
            'c_train': c_train,
            'y_train': y_train,
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
            'c_emb_train': c_emb_train,
            'c_emb_test': c_emb_test,
            'y_emb_train': y2_train,
            'y_emb_test': y2_test,
            'train_acc_bool': model_1.loss_list,
            'train_acc_fuzzy': model_3.loss_list,
            'train_acc_emb': model_2.loss_list,
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
            Linear(n_features, 500),
            LeakyReLU(),
            Linear(500, n_concepts),
            LeakyReLU(),
            ConceptEmbeddings(in_features=n_concepts, out_features=n_concepts, emb_size=5, bias=True),
            Sigmoid()
        ])
        # self.c2y_model = NeSyLayer(5, n_concepts, n_concepts, n_tasks)
        self.c2y_model = Sequential(*[
            Linear(n_concepts, 250),
            LeakyReLU(),
            Linear(250, 250),
            LeakyReLU(),
            Linear(250, n_tasks),
        ])
        self.loss = BCELoss()
        self.loss2 = CrossEntropyLoss()
        self.loss_list = []

    def forward(self, batch):
        x, y, c = batch
        c = self.x2c_model(x)
        # y = self.c2y_model(to_boolean(c, 2, 1))
        y = self.c2y_model(to_boolean(c, 2, 1).permute(0, 2, 1)).permute(0, 2, 1)
        c_sem = semantics(c)
        # y_sem = semantics(y)
        y_sem = torch.exp(-torch.norm(y, p=2, dim=-1))
        return c_sem, y_sem, c

    def training_step(self, batch, batch_no):
        x, y, c = batch
        c_sem, y_sem, _ = self(batch)
        loss = self.loss(c_sem, c) + 0.1*self.loss2(y_sem, y)
        c_accuracy, y_accuracy = compute_accuracy(c_sem, y_sem, c, y)
        print(f'Loss {loss:.4f}, c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}, ')
        self.loss_list.append([c_accuracy, y_accuracy])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.008)


class TrigoNet(pl.LightningModule):
    def __init__(self, n_features, n_concepts, n_tasks, bool):
        super().__init__()
        self.n_concepts = n_concepts
        self.x2c_model = Sequential(*[
            Linear(n_features, 500),
            LeakyReLU(),
            Linear(500, n_concepts),
            LeakyReLU(),
            Linear(in_features=n_concepts, out_features=n_concepts, bias=True),
            Sigmoid()
        ])
        self.c2y_model = Sequential(*[
            Linear(n_concepts, 250),
            LeakyReLU(),
            Linear(250, 250),
            LeakyReLU(),
            Linear(250, n_tasks),
            Softmax(dim=-1)
        ])
        self.loss = BCELoss()
        self.loss2 = CrossEntropyLoss()
        self.bool = bool
        self.loss_list = []

    def forward(self, batch):
        x, y, c = batch
        c = self.x2c_model(x)
        if self.bool:
            y = self.c2y_model((c>0.5).float())
        else:
            y = self.c2y_model(c)
        return c, y

    def training_step(self, batch, batch_no):
        x, c, y = batch
        c_logits, y_logits = self(batch)
        loss = self.loss(c_logits, c) + 0.1*self.loss2(y_logits, y)
        # compute accuracy
        c_accuracy, y_accuracy = compute_accuracy(c_logits, y_logits, c, y)
        print(f'Loss {loss:.4f}, c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}, ')
        self.loss_list.append([c_accuracy, y_accuracy])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.008)


def compute_accuracy(c_pred, y_pred, c_true, y_true):
    c_pred = c_pred.ravel().cpu().detach() > 0.5
    y_pred = y_pred.argmax(dim=-1).cpu().detach()
    c_true = c_true.ravel().cpu().detach()
    y_true = y_true.ravel().cpu().detach()
    c_accuracy = accuracy_score(c_true, c_pred)
    y_accuracy = accuracy_score(y_true, y_pred)
    return c_accuracy, y_accuracy


if __name__ == '__main__':
    main()
