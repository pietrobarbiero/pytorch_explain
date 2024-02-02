import sys, os
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

from experiments.data.load_datasets import load_mnist_addition, extract_image_features
from torch_explain import datasets
from torch_explain.nn.concepts import ConceptLinearLayer
from torch_explain.nn.semantics import GodelTNorm, ProductTNorm

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def train_concept_bottleneck_model(train_dataset, test_dataset, extracting_model='resnet18', embedding_size=32,
                                   encoder_size=128, concept_names=None, epochs=1000, lr=0.001,
                                   weight_task=0.1, weight_reg=1e-5, bias_reg=1e-2, bias=False, load=True):

    x_train, c_train, y_train = extract_image_features(train_dataset, extracting_model, load=load,
                                                       filename=f'data/mnist_addition_{extracting_model}_train.pt')
    x_test, c_test, y_test = extract_image_features(test_dataset, extracting_model, load=load,
                                                    filename=f'data/mnist_addition_{extracting_model}_test.pt')

    x_train, c_train, y_train = x_train.to(device), c_train.to(device), y_train.to(device)
    x_test, c_test, y_test = x_test.to(device), c_test.to(device), y_test.to(device)

    # x_train, c_train, y_train = x_train[:10000], c_train[:10000], y_train[:10000]

    print("X", x_train.shape, "C", c_train.shape, "Y", y_train.shape)
    encoder = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], encoder_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(encoder_size, encoder_size),
        torch.nn.LeakyReLU(),
    )
    concept_embedder = te.nn.ConceptEmbedding(encoder_size, c_train.shape[1], embedding_size)
    task_predictor = ConceptLinearLayer(embedding_size, 19, bias=bias)
    model = torch.nn.Sequential(encoder, concept_embedder, task_predictor)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_form_c = torch.nn.BCELoss()
    loss_form_y = torch.nn.CrossEntropyLoss()
    pbar = tqdm(range(epochs), desc='Training', ncols=200)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        h = encoder(x_train)
        c_emb, c_pred = concept_embedder(h, c=c_train, train=True)
        if bias:
            y_pred, weight_attn, bias_attn = task_predictor(c_emb, c_pred, return_attn=True)
        else:
            y_pred, weight_attn = task_predictor(c_emb, c_pred, return_attn=True)

        concept_loss = loss_form_c(c_pred, c_train)
        task_loss = loss_form_y(y_pred, y_train)
        weight_attn_reg = task_predictor.entropy_reg(weight_attn)

        if bias:
            loss = concept_loss + weight_task * task_loss + \
                   weight_reg * weight_attn_reg + bias_reg * torch.mean(bias_attn.abs() ** 2)
        else:
            loss = concept_loss + weight_task * task_loss + weight_reg * weight_attn_reg
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                h = encoder(x_test)
                c_emb, c_pred = concept_embedder(h)
                y_pred = task_predictor(c_emb, c_pred)

                task_acc = f1_score(y_test.cpu(), y_pred.cpu().argmax(dim=1), average='macro')
                concept_acc = f1_score(c_test.cpu(), c_pred.cpu() > 0.5, average='macro')
                post_fix = f'l {loss.item():.3f} #tf1: {task_acc:.3f} '\
                           f'cf1: {concept_acc:.3f} wl: {weight_attn.abs().mean().item():.3f} '
                if bias:
                    post_fix += f'bl: {bias_attn.abs().mean().item():.3f}'
                pbar.set_postfix_str(post_fix)

    # local_explanations = task_predictor.explain(c_emb, c_pred, 'local', concept_names)
    global_explanations = task_predictor.explain(c_emb, c_pred, 'global', concept_names)
    for global_explanations_i in global_explanations:
        print(global_explanations_i)

    return model


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)

    train_ds, test_ds = load_mnist_addition()

    train_concept_bottleneck_model(train_ds, test_ds,
                                   concept_names=['Zero1', 'One1', 'Two1', 'Three1', 'Four1', 'Five1', 'Six1', 'Seven1',
                                                  'Eight1', 'Nine1',
                                                  'Zero2', 'One2', 'Two2', 'Three2', 'Four2', 'Five2', 'Six2', 'Seven2',
                                                  'Eight2', 'Nine2'])

