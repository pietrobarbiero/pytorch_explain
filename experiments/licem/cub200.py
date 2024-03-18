import sys, os

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

from experiments.data.load_datasets import load_cub_v2
from experiments.data.utils import extract_image_features, show_batch

from torch_explain import datasets
from torch_explain.nn.concepts import ConceptLinearLayer
from torch_explain.nn.semantics import GodelTNorm, ProductTNorm

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# data_folder = "/data1" if torch.cuda.is_available() else "data"
data_folder = "data"


def train_cub(extracting_model='resnet18', embedding_size=32,
                                   encoder_size=512, epochs=1000, lr=0.001, batch_size=None,
                                   weight_task=0.1, weight_reg=1e-5, bias_reg=1e-5, bias=False, data_aug=True, load=True):

    train_dataset, test_dataset = load_cub_v2()
    concept_names = train_dataset.dataset.attribute_names

    show_batch(train_dataset, labels_names=concept_names, visualize_concept=0)
    show_batch(test_dataset, labels_names=concept_names, visualize_concept=0)

    x_train, c_train, y_train = extract_image_features(train_dataset, extracting_model, load=load, data_augmentation=data_aug,
                                                       filename=f'{data_folder}/cub_{extracting_model}_daug_{data_aug}_train.pt')
    x_test, c_test, y_test = extract_image_features(test_dataset, extracting_model, load=load,
                                                    filename=f'{data_folder}/cub_{extracting_model}_test.pt')

    test_dl = DataLoader(TensorDataset(x_test, c_test, y_test),
                         batch_size=batch_size*5, shuffle=False)

    # x_train, c_train, y_train = x_train.to(device), c_train.to(device), y_train.to(device)
    # x_test, c_test, y_test = x_test.to(device), c_test.to(device), y_test.to(device)

    # x_train, c_train, y_train = x_train[:10000], c_train[:10000], y_train[:10000]
    print("Train: X", x_train.shape, "C", c_train.shape, "Y", y_train.shape)
    print("Test: X", x_test.shape, "C", c_test.shape, "Y", y_test.shape)

    if 'resnet' in extracting_model:
        encoder = torch.nn.Identity()
        encoder_output_size = x_train.shape[1]
    else:
        encoder = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        encoder_output_size = encoder.fc.in_features
        encoder.fc = torch.nn.Identity()

    concept_encoder = torch.nn.Sequential(
        torch.nn.Linear(encoder_output_size, encoder_size),
        # torch.nn.LeakyReLU(),
        # torch.nn.Linear(encoder_size, encoder_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(encoder_size, c_train.shape[1]),
        # torch.nn.Sigmoid()
    )

    # concept_embedder = te.nn.ConceptEmbedding(encoder_size, c_train.shape[1], embedding_size)
    # task_predictor = ConceptLinearLayer(embedding_size, y_train.max()+1, bias=bias)
    task_predictor = torch.nn.Linear(c_train.shape[1], y_train.max()+1)
    model = torch.nn.Sequential(encoder, concept_encoder, task_predictor)#, concept_embedder, task_predictor)
    model.to(device)

    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters(), 'lr': lr*1e-2},
        {'params': concept_encoder.parameters()},
        {'params': task_predictor.parameters()}
    ], lr=lr)

    loss_form_c = torch.nn.BCEWithLogitsLoss()
    loss_form_y = torch.nn.CrossEntropyLoss()
    pbar = tqdm(range(epochs), desc='Training', ncols=150)
    for epoch in pbar:
        model.train()

        c_preds, y_preds = [], []
        c_labels, y_labels = [], []
        if batch_size is not None:
            loader = torch.utils.data.DataLoader(TensorDataset(x_train, c_train, y_train),
                                                 batch_size=batch_size, shuffle=True)
        else:
            loader = [(x_train, c_train, y_train)]
        for x_batch, c_batch, y_batch in loader:
            x_batch, c_batch, y_batch = x_batch.to(device), c_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            h = encoder(x_batch)
            c_pred = concept_encoder(h)
            # c_emb, c_pred = concept_embedder(h, c=c_batch, train=True)
            # if bias:
            #     y_pred, weight_attn, bias_attn = task_predictor(c_emb, c_pred, return_attn=True)
            # else:
            #     y_pred, weight_attn = task_predictor(c_emb, c_pred, return_attn=True)
            y_pred = task_predictor(c_pred)

            concept_loss = loss_form_c(c_pred, c_batch)
            task_loss = loss_form_y(y_pred, y_batch)
            loss = concept_loss + weight_task * task_loss

            # if bias:
            #     loss = concept_loss + weight_task * task_loss + \
            #            weight_reg * weight_attn_reg + bias_reg * torch.mean(bias_attn.abs() ** 2)
            # else:
            #     loss = concept_loss + weight_task * task_loss + weight_reg * weight_attn_reg
            loss.backward()
            optimizer.step()

            c_preds.append(c_pred)
            c_labels.append(c_batch)
            y_preds.append(y_pred)
            y_labels.append(y_batch)

        c_preds = torch.cat(c_preds, dim=0)
        c_labels = torch.cat(c_labels, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        y_labels = torch.cat(y_labels, dim=0)

        # compute accuracy
        if (epoch) % 10 == 0:
            with torch.no_grad():
                model.eval()
                c_pred_test, c_test, y_test = [], [], []
                for data in test_dl:
                    c_pred_test.append(concept_encoder(encoder(data[0].to(device))))
                    c_test.append(data[1]), y_test.append(data[2])

                c_pred_test = torch.cat(c_pred_test, dim=0)
                c_test, y_test = torch.cat(c_test, dim=0), torch.cat(y_test, dim=0)

                # y_pred = task_predictor(c_emb, c_pred)
                y_pred_test = task_predictor(c_pred_test)

                # concept_acc = f1_score(c_test.cpu(), c_pred.cpu() > 0.5, average='macro')
                # post_fix = f'l {loss.item():.3f} #tf1: {task_acc:.3f} '\
                #            f'cf1: {concept_acc:.3f} wl: {weight_attn.abs().mean().item():.3f} '
                # if bias:
                #     post_fix += f'bl: {bias_attn.abs().mean().item():.3f}'
                c_test_f1 = f1_score(c_test.cpu(), c_pred_test.cpu() > 0.5, average='macro')
                c_test_acc = (c_test.cpu() == (c_pred_test.cpu() > 0.5)).sum()/(c_test.shape[0]*c_test.shape[1])
                c_train_f1 = f1_score(c_labels.cpu(), c_preds.cpu() > 0.5, average='macro')
                # y_train_acc = f1_score(y_batch.cpu(), y_pred.cpu().argmax(dim=1), average='macro')
                # y_test_acc = f1_score(y_test.cpu(), y_pred_test.cpu().argmax(dim=1), average='macro')
                y_train_acc = accuracy_score(y_labels.cpu(), y_preds.cpu().argmax(dim=1))
                y_test_acc = accuracy_score(y_test.cpu(), y_pred_test.cpu().argmax(dim=1))
                print("")

        post_fix = f'l {concept_loss.item():.3f} y_train: {y_train_acc:.3f} y_test: {y_test_acc:.3f} '\
                   f'c_test_f1: {c_test_f1:.3f} c_test_acc: {c_test_acc:.3f} c_train_f1: {c_train_f1:.3f}'
        pbar.set_postfix_str(post_fix)

    # local_explanations = task_predictor.explain(c_emb, c_pred, 'local', concept_names)
    # global_explanations = task_predictor.explain(c_emb, c_pred, 'global', concept_names)
    # for global_explanations_i in global_explanations:
    #     print(global_explanations_i)

    return model


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)

    train_cub(lr=0.0003, epochs=5001, data_aug=False, weight_task=0.0,
              extracting_model='', batch_size=10000, load=False)
    train_cub(lr=0.0003, epochs=10001, data_aug=True, weight_task=0.0,
              extracting_model='', batch_size=10000, load=False)

    # train_cub(lr=0.0003, epochs=5001, data_aug=False, weight_task=0.0,
    #                                extracting_model='resnet50', batch_size=10000)
    # train_cub(lr=0.0003, epochs=10001, data_aug=True, weight_task=0.0,
    #                                extracting_model='resnet50', batch_size=10000)

