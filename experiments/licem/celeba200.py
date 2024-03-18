import sys, os

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

sys.path.append("../../")

import torch_explain as te
import unittest

import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm

from experiments.data.load_datasets import load_celeba
from experiments.data.utils import extract_image_features, show_batch

from torch_explain import datasets
from torch_explain.nn.concepts import ConceptLinearLayer, ConceptReasoningLayer
from torch_explain.nn.semantics import GodelTNorm, ProductTNorm

torch.manual_seed(0)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# data_folder = "/data1" if torch.cuda.is_available() else "data"
data_folder = "data"
print("Data folder:", data_folder)


def train_celeba(name='cbm', extracting_model='resnet18', embedding_size=32,
                 encoder_size=512, epochs=100, lr=0.001, batch_size=None,
                 weight_task=0.1, weight_reg=1e-5, bias_reg=1e-5, bias=False, data_aug=False, load=True,
                 n_train_samples=None, simplified=False, concept_embedding=False, task_predictor='bb'):
    print("Task Weight:", weight_task)
    print("Extracting model:", extracting_model)
    print("Data augmentation:", data_aug)
    print("Simplified:", simplified)
    print("N train samples:", n_train_samples)
    print("LR:", lr)
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Embedding size:", embedding_size)

    train_dataset, test_dataset = load_celeba(base_dir=data_folder)
    concept_names = train_dataset.dataset.attr_names[:20]
    label_names = train_dataset.dataset.attr_names[20:]
    c_idx = concept_names.index("Blond_Hair")

    # show_batch(train_dataset, labels_names=concept_names, visualize_concept=c_idx)
    # show_batch(test_dataset, labels_names=concept_names, visualize_concept=c_idx)

    x_train, c_train, y_train = extract_image_features(train_dataset, extracting_model, load=load, data_augmentation=data_aug,
                                                       filename=f'{data_folder}/celeba_{extracting_model}_daug_{data_aug}_train.pt')
    x_test, c_test, y_test = extract_image_features(test_dataset, extracting_model, load=load,
                                                    filename=f'{data_folder}/celeba_{extracting_model}_test.pt')

    c_train, c_test, y_train, y_test = c_train.to(float), c_test.to(float), y_train.to(float), y_test.to(float)

    if simplified:
        c_train = y_train[:, 0:1]
        c_test = y_test[:, 0:1]
        y_train = 1 - y_train[:, 0:1]
        y_test = 1 - y_test[:, 0:1]

    if n_train_samples is not None:
        x_train, c_train, y_train = x_train[:n_train_samples], c_train[:n_train_samples], y_train[:n_train_samples]

    print("Train: X", x_train.shape, "C", c_train.shape, "Y", y_train.shape)
    print("Test: X", x_test.shape, "C", c_test.shape, "Y", y_test.shape)

    if 'resnet' in extracting_model:
        encoder = torch.nn.Identity()
        encoder_output_size = x_train.shape[1]
    else:
        encoder = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        encoder_output_size = encoder.fc.in_features
        encoder.fc = torch.nn.Identity()

    if concept_embedding:
        concept_encoder = te.nn.ConceptEmbedding(encoder_output_size, c_train.shape[1], embedding_size)
        c_size = embedding_size
    else:
        concept_encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_size, encoder_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(encoder_size, c_train.shape[1]),
        )
        c_size = c_train.shape[1]

    if task_predictor == 'bb':
        task_predictor = torch.nn.Sequential(
            torch.nn.Linear(c_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, y_train.shape[1]),
        )
    elif task_predictor == 'linear':
        assert concept_embedding, "Concept embedding must be True for ConceptLinearLayer"
        task_predictor = ConceptLinearLayer(embedding_size, y_train.max()+1, bias=bias)
    elif task_predictor == 'dcr':
        assert concept_embedding, "Concept embedding must be True for ConceptLinearLayer"
        task_predictor = ConceptReasoningLayer(embedding_size, y_train.max()+1, temperature=1.)

    model = torch.nn.Sequential(encoder, concept_encoder, task_predictor)
    model.to(device)
    print(model)

    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters(), 'lr': lr*1e-2},
        {'params': concept_encoder.parameters()},
        {'params': task_predictor.parameters()}
    ], lr=lr)

    loss_form_c = torch.nn.BCEWithLogitsLoss()
    loss_form_y = torch.nn.BCEWithLogitsLoss()
    pbar = tqdm(range(epochs), desc='Training', ncols=150)

    if batch_size is not None:
        train_dl = torch.utils.data.DataLoader(TensorDataset(x_train, c_train, y_train),
                                               batch_size=batch_size, shuffle=True, )
        test_dl = DataLoader(TensorDataset(x_test, c_test, y_test),
                             batch_size=batch_size * 5, shuffle=False)
    else:
        train_dl = [(x_train, c_train, y_train)]
        test_dl = [(x_test, c_test, y_test)]

    for epoch in pbar:
        model.train()

        c_preds, y_preds = [], []
        c_labels, y_labels = [], []
        losses = 0.
        for x_batch, c_batch, y_batch in train_dl:
            x_batch, c_batch, y_batch = x_batch.to(device), c_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            h = encoder(x_batch)
            if concept_embedding:
                c_emb, c_pred = concept_encoder(h, c=c_batch, train=True)
                y_pred = task_predictor(c_pred)

            else:
                c_pred = concept_encoder(h)
                y_pred = task_predictor(c_pred)

            concept_loss = loss_form_c(c_pred, c_batch)
            task_loss = loss_form_y(y_pred, y_batch)
            loss = concept_loss + weight_task * task_loss

            loss.backward()
            optimizer.step()

            c_preds.append(c_pred.detach())
            c_labels.append(c_batch)
            y_preds.append(y_pred.detach())
            y_labels.append(y_batch)
            losses += loss.item()

        c_preds = torch.cat(c_preds, dim=0)
        c_labels = torch.cat(c_labels, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        y_labels = torch.cat(y_labels, dim=0)
        losses = losses / len(train_dl)

        # compute accuracy
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                c_pred_test = []
                for data in test_dl:
                    h = encoder(data[0].to(device))
                    if concept_embedding:
                        c_emb, c_pred = concept_encoder(h)
                        c_pred_test.append(c_pred)
                    else:
                        c_pred = concept_encoder(h)
                        c_pred_test.append(c_pred)

                c_pred_test = torch.cat(c_pred_test, dim=0)
                y_pred_test = task_predictor(c_pred_test)

                c_test_f1 = f1_score(c_pred_test.cpu() > 0., c_test.cpu(),  average='macro')
                c_test_acc = (c_test.cpu() == (c_pred_test.cpu() > 0.)).sum()/(c_test.shape[0]*c_test.shape[1])
                c_train_f1 = f1_score(c_preds.cpu() > 0., c_labels.cpu(),  average='macro')
                y_train_acc = f1_score(y_preds.cpu() > 0., y_labels.cpu(), average='macro')
                y_test_acc = f1_score(y_pred_test.cpu() > 0., y_test.cpu(), average='macro')

        post_fix = f'l {losses:.3f} y_train: {y_train_acc:.3f} y_test: {y_test_acc:.3f} '\
                   f'c_test_f1: {c_test_f1:.3f} c_test_acc: {c_test_acc:.3f} c_train_f1: {c_train_f1:.3f}'
        pbar.set_postfix_str(post_fix)
        print("")

    global_explanations = []
    if task_predictor == 'Linear' or task_predictor == 'dcr':
        global_explanations = task_predictor.explain(c_emb, c_pred, 'global', concept_names)
        for global_explanations_i in global_explanations:
            print(global_explanations_i)

    print(f'{name} - c_test_f1: {c_test_f1:.3f} c_test_acc: {c_test_acc:.3f} c_train_f1: {c_train_f1:.3f} '
            f'y_train_acc: {y_train_acc:.3f} y_test_acc: {y_test_acc:.3f}')

    # create dataframe with results
    df = pd.DataFrame([{
        'name': name,
        'extracting_model': extracting_model,
        'data_aug': data_aug,
        'simplified': simplified,
        'n_train_samples': n_train_samples,
        'lr': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'embedding_size': embedding_size,
        'concept_embedding': concept_embedding,
        'task_predictor': task_predictor,
        'c_train_f1': c_train_f1,
        'c_test_f1': c_test_f1,
        'c_test_acc': c_test_acc,
        'y_train_acc': y_train_acc,
        'y_test_acc': y_test_acc,
    }])

    return df


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)

    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # CBM
    # df1 = train_celeba(name='cbm', extracting_model="resnet50")
    # df1.to_csv(f'{results_folder}/celeba_cbm.csv', index=False)

    # CEM
    df2 = train_celeba(name='cem', extracting_model="resnet50", concept_embedding=True)
    df2.to_csv(f'{results_folder}/celeba_cem.csv', index=False)

    # DCR
    df3 = train_celeba(name='dcr', extracting_model="resnet50", concept_embedding=True, task_predictor='dcr')
    df3.to_csv(f'{results_folder}/celeba_dcr.csv', index=False)

    # Linear
    df4 = train_celeba(name='linear', extracting_model="resnet50", concept_embedding=True, task_predictor='linear')
    df4.to_csv(f'{results_folder}/celeba_linear.csv', index=False)

    dfs = pd.concat([df1, df2, df3, df4])
    dfs.to_csv(f'{results_folder}/celeba.csv', index=False)
    print(dfs)
    print("\n Done \n")


