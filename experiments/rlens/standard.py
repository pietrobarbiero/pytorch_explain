import json
import pickle

import joblib
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import torch
from torch.nn import Module
from torch_explain.logic.nn import entropy
from torch_geometric.nn import Sequential
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import joblib

import torch_explain as te
from torch.nn.functional import one_hot


class NN(Module):
    def __init__(self, lr=0.008):
        super().__init__()
        self.lr = lr

    def fit(self, x, y):
        x = torch.FloatTensor(x)
        y = one_hot(torch.LongTensor(y)).float()

        self.layers = [
            te.nn.EntropyLinear(x.shape[1], 50, n_classes=y.shape[1], temperature=10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 1),
        ]
        self.model = torch.nn.Sequential(*self.layers)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        loss_form = torch.nn.BCEWithLogitsLoss()
        self.model.train()

        concept_names = [f'x{i}' for i in range(x.shape[1])]
        class_names = [f'y{i}' for i in range(y.shape[1])]
        np.random.seed(42)
        train_mask = set(np.random.choice(np.arange(x.shape[0]), int(x.shape[0] * 0.8), replace=False))
        test_mask = set(np.arange(x.shape[0])) - train_mask
        self.train_mask = torch.LongTensor(list(train_mask))
        self.test_mask = torch.LongTensor(list(test_mask))
        for epoch in range(5001):
            # train step
            optimizer.zero_grad()
            y_pred = self.forward(x).squeeze(-1)
            loss = loss_form(y_pred[self.train_mask], y[self.train_mask])
            loss.backward()
            optimizer.step()

            # compute accuracy
            if epoch % 100 == 0:
                train_accuracy = (y_pred[self.train_mask] > 0.5).eq(y[self.train_mask]).sum().item() / (y[self.train_mask].size(0) * y[self.train_mask].size(1))
                test_accuracy = (y_pred[self.test_mask] > 0.5).eq(y[self.test_mask]).sum().item() / (y[self.test_mask].size(0) * y[self.test_mask].size(1))
                print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {train_accuracy:.4f} test accuracy: {test_accuracy:.4f}')

        self.clusters_bool_ = torch.unique(self.clusters_ > 0.5, dim=0).float()
        a = torch.cdist(self.clusters_.permute(1, 0, 2), self.clusters_bool_.permute(1, 0, 2))
        self.centroids = torch.argmin(a, dim=1)
        self.feature_selected_ = self.model._modules['0'].alpha_norm > 0.5

        explanations, local_exp = entropy.explain_classes(self.model, x, y, self.train_mask, self.test_mask,
                                                          c_threshold=0.5, y_threshold=0., verbose=True,
                                                          concept_names=concept_names, class_names=class_names,
                                                          material=True, good_bad_terms=False, max_accuracy=True)
        return explanations


    def forward(self, x):
        self.concepts_ = self.layers[0](x)
        h = self.layers[1](self.concepts_)
        h = self.layers[2](h)
        self.clusters_ = self.layers[3](h)
        y = self.layers[4](self.clusters_)
        return y


def main():
    datasets = ['xor', 'trig', 'vec']
    # datasets = ['trig']
    folds = [i+1 for i in range(5)]
    epoch = 500
    for dataset in datasets:
        results = pd.DataFrame()
        for fold in folds:
            c_test = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_embedding_semantics_on_epoch_{epoch}.npy')
            # c1 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/c_test.npy')
            # c2 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/c_val.npy')
            # y1 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/y_test.npy')
            y_test = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/y_val.npy')

            model = NN()
            explanations = model.fit(c_test, y_test)

            res_dir = f'./results/{dataset}_activations_final_rerun/explanations/'
            os.makedirs(res_dir, exist_ok=True)
            out_file = os.path.join(res_dir, 'explanations.csv')
            res = pd.DataFrame(explanations).T
            res['split'] = fold

            if len(results) == 0:
                results = res
            else:
                results = pd.concat((results, res))

            results.to_csv(out_file)

    # tsne = TSNE(n_components=2, random_state=42)
    # x2 = tsne.fit_transform(c_test)
    #
    # plt.figure()
    # plt.scatter(x2[:, 0], x2[:, 1], marker='o', c='k', alpha=0.1, label='samples')
    # plt.scatter(x2[model.centroids[0], 0], x2[model.centroids[0], 1], marker='x', c='r', label='centroids')
    # plt.scatter(x2[model.centroids[1], 0], x2[model.centroids[1], 1], marker='x', c='r', label=None)
    # plt.scatter(x2[model.centroids[2], 0], x2[model.centroids[2], 1], marker='x', c='r', label=None)
    # plt.legend()
    # plt.savefig('clustering.png')
    # plt.show()

    # y1 = y.argmax(dim=1)
    # plt.figure()
    # plt.scatter(x2[y1 == 0, 0], x2[y1 == 0, 1], marker='o', c='r', alpha=0.1)
    # plt.scatter(x2[y1 == 1, 0], x2[y1 == 1, 1], marker='o', c='b', alpha=0.1)
    # plt.scatter(x2[y1 == 2, 0], x2[y1 == 2, 1], marker='o', c='g', alpha=0.1)
    # plt.scatter(x2[model.centroids[0], 0], x2[model.centroids[0], 1], marker='x', c='r')
    # plt.scatter(x2[model.centroids[1], 0], x2[model.centroids[1], 1], marker='x', c='b')
    # plt.scatter(x2[model.centroids[2], 0], x2[model.centroids[2], 1], marker='x', c='g')
    # plt.show()
    # plt.figure()
    # plt.scatter(x2[:, 0], x2[:, 1], marker='o', c='k', alpha=0.1)
    # plt.scatter(x2[y1 == 0, 0], x2[y1 == 0, 1], marker='o', c='g', alpha=0.1)
    # plt.scatter(x2[self.centroids[0], 0], x2[self.centroids[0], 1], marker='x', c='r')
    # plt.show()

if __name__ == '__main__':
    main()
