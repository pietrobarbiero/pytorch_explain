import unittest

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm

import torch_explain as te
from experiments.data.load_datasets import load_mnist_addition, extract_image_features
from torch_explain import datasets
from torch_explain.nn.concepts import ConceptLinearLayer
from torch_explain.nn.semantics import GodelTNorm, ProductTNorm


def train_concept_bottleneck_model(train_dataset, test_dataset, embedding_size=32, concept_names=None):

    # extract_features
    x_train, c_train, y_train = extract_image_features(train_dataset, filename='data/mnist_addition_train.pt')
    x_test, c_test, y_test = extract_image_features(test_dataset, filename='data/mnist_addition_test.pt')

    y_train = F.one_hot(y_train.long().ravel()).float()
    y_test = F.one_hot(y_test.long().ravel()).float()

    print("X", x_train.shape, "C", c_train.shape, "Y", y_train.shape)

    # concept embedding model
    # encoder = torch.nn.Sequential(
    #     torch.nn.Linear(x_train.shape[1], 100),
    #     torch.nn.LeakyReLU(),
    # )
    concept_embedder = te.nn.ConceptEmbedding(x_train.shape[1], c_train.shape[1], embedding_size)
    task_predictor = ConceptLinearLayer(embedding_size, 19, bias=True)
    model = torch.nn.Sequential(concept_embedder, task_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_form_c = torch.nn.BCELoss()
    loss_form_y = torch.nn.BCELoss()
    model.train()
    for epoch in pbar := tqdm(range(200), desc='Training'):
        optimizer.zero_grad()

        c_emb, c_pred = concept_embedder(x_train, c=c_train, train=True)
        y_pred, weight_attn, bias_attn = task_predictor(c_emb, c_pred, return_attn=True)

        concept_loss = loss_form_c(c_pred, c_train)
        task_loss = loss_form_y(y_pred, y_train)
        # attn_weight_reg = torch.mean(weight_attn**2)
        weight_attn_reg = task_predictor.entropy_reg(weight_attn)
        loss = concept_loss + 0.5*task_loss + 1e-4*weight_attn_reg + 1e-4*torch.mean(bias_attn.abs()**2)
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 1 == 0:
            c_emb, c_pred = concept_embedder(x_test)
            y_pred = task_predictor(c_emb, c_pred)

            task_accuracy = accuracy_score(y_test, y_pred > 0.5)
            concept_accuracy = accuracy_score(c_test, c_pred > 0.5)
            pbar.set_postfix_str(f'Epoch {epoch}: loss {loss:.4f} task accuracy: {task_accuracy:.4f} '
                  f'concept accuracy: {concept_accuracy:.4f} weight loss: {weight_attn:.4f} bias loss: {torch.mean(bias_attn.abs()):.4f} ')

    # local_explanations = task_predictor.explain(c_emb, c_pred, 'local', concept_names)
    global_explanations = task_predictor.explain(c_emb, c_pred, 'global', concept_names)
    print(global_explanations)

    return model


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)

    train_dataset, test_dataset = load_mnist_addition()

    train_concept_bottleneck_model(train_dataset, test_dataset, embedding_size=32, concept_names=['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine',
                                                                                                  'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'])

