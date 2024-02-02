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


def train_concept_bottleneck_model(train_dataset, test_dataset, extracting_model='resnet18', embedding_size=32, concept_names=None):


    x_train, c_train, y_train = extract_image_features(train_dataset, extracting_model,
                                                       filename=f'data/mnist_addition_{extracting_model}_train.pt')
    x_test, c_test, y_test = extract_image_features(test_dataset, extracting_model,
                                                    filename=f'data/mnist_addition_{extracting_model}_test.pt')

    # y_train = F.one_hot(y_train.long().ravel()).float()
    # y_test = F.one_hot(y_test.long().ravel()).float()

    x_train, c_train, y_train = x_train[:10000], c_train[:10000], y_train[:10000]

    print("X", x_train.shape, "C", c_train.shape, "Y", y_train.shape)
    encoder_size = embedding_size * 2
    encoder = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], encoder_size),
        torch.nn.LeakyReLU(),
    )
    concept_embedder = te.nn.ConceptEmbedding(encoder_size, c_train.shape[1], embedding_size)
    task_predictor = ConceptLinearLayer(embedding_size, 19, bias=True)
    model = torch.nn.Sequential(encoder, concept_embedder, task_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form_c = torch.nn.BCELoss()
    loss_form_y = torch.nn.CrossEntropyLoss()
    model.train()
    pbar = tqdm(range(500), desc='Training')
    for epoch in pbar:
        optimizer.zero_grad()

        h = encoder(x_train)
        c_emb, c_pred = concept_embedder(h, c=c_train, train=True)
        y_pred, weight_attn, bias_attn = task_predictor(c_emb, c_pred, return_attn=True)

        concept_loss = loss_form_c(c_pred, c_train)
        task_loss = loss_form_y(y_pred, y_train)
        weight_attn_reg = task_predictor.entropy_reg(weight_attn)
        loss = concept_loss + 0.1*task_loss + 1e-5*weight_attn_reg + 1e-5*torch.mean(bias_attn.abs()**2)
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 1 == 0:
            with torch.no_grad():
                h = encoder(x_test)
                c_emb, c_pred = concept_embedder(h)
                y_pred = task_predictor(c_emb, c_pred)

                task_accuracy = f1_score(y_test, y_pred.argmax(dim=1), average='macro')
                concept_accuracy = f1_score(c_test, c_pred > 0.5, average='macro')
                post_fix = f'Epoch {epoch}: loss {loss.item():.4f} task acc: {task_accuracy:.4f} '\
                           f'concept acc: {concept_accuracy:.4f} weight l: {weight_attn.abs().mean().item():.4f} '\
                           f'bias l: {torch.mean(bias_attn.abs()):.4f}'
                pbar.set_postfix_str(post_fix)

    # local_explanations = task_predictor.explain(c_emb, c_pred, 'local', concept_names)
    global_explanations = task_predictor.explain(c_emb, c_pred, 'global', concept_names)
    print(global_explanations)

    return model


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)

    train_dataset, test_dataset = load_mnist_addition()

    train_concept_bottleneck_model(train_dataset, test_dataset,
                                   concept_names=['Zero1', 'One1', 'Two1', 'Three1', 'Four1', 'Five1', 'Six1', 'Seven1', 'Eight1', 'Nine1',
                                                  'Zero2', 'One2', 'Two2', 'Three2', 'Four2', 'Five2', 'Six2', 'Seven2', 'Eight2', 'Nine2'])

