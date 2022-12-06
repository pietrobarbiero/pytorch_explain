from sklearn.datasets import load_iris
import numpy as np
import torch
from torch.nn import Module
from torch_geometric.nn import Sequential
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

import torch_explain as te
from torch.nn.functional import one_hot


class NN(Module):
    def __init__(self, in_features, out_features, emb_size):
        super().__init__()
        self.entropy = te.nn.EntropyLinear(in_features, 20, n_classes=out_features, temperature=0.3)
        self.linear = torch.nn.Linear(20, 1)
        self.scoring = torch.nn.Linear(emb_size, 1)

    def forward(self, x):
        self.concepts_ = torch.nn.functional.leaky_relu(self.entropy(x))
        self.concepts_reshaped_ = self.concepts_.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        self.clusters_ = torch.nn.functional.leaky_relu(self.linear(self.concepts_))
        self.clusters_reshaped_ = self.clusters_.reshape(x.shape[0], x.shape[1], x.shape[2])
        y = self.scoring(self.clusters_reshaped_.reshape(x.shape[0]*x.shape[1], -1))
        y = y.reshape(x.shape[0], -1)
        return y


def fit(x, y, lr=0.001):
    x = torch.FloatTensor(x)
    y = one_hot(torch.LongTensor(y)).float()
    model = NN(x.shape[1], y.shape[1], x.shape[2])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_form = torch.nn.BCEWithLogitsLoss()
    model.train()

    concept_names = [f'x{i}' for i in range(x.shape[1])]
    class_names = [f'y{i}' for i in range(y.shape[1])]
    np.random.seed(42)
    train_mask = set(np.random.choice(np.arange(x.shape[0]), int(x.shape[0] * 0.8), replace=False))
    test_mask = set(np.arange(x.shape[0])) - train_mask
    train_mask = torch.LongTensor(list(train_mask))
    test_mask = torch.LongTensor(list(test_mask))
    for epoch in range(2001):
        # train step
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_form(y_pred[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 100 == 0:
            train_accuracy = (y_pred[train_mask] > 0.).eq(y[train_mask]).sum().item() / (y[train_mask].size(0) * y[train_mask].size(1))
            test_accuracy = (y_pred[test_mask] > 0.).eq(y[test_mask]).sum().item() / (y[test_mask].size(0) * y[test_mask].size(1))
            print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {train_accuracy:.4f} test accuracy: {test_accuracy:.4f}')

    clusters_bool_ = torch.unique(model.clusters_reshaped_ > 0.5, dim=0).float()
    a = torch.cdist(model.clusters_.permute(1, 0, 2), model.clusters_bool_.permute(1, 0, 2))
    centroids = torch.argmin(a, dim=1)
    feature_selected_ = model.entropy.alpha_norm > 0.5
    return



def main():
    c_test = np.load('./results/xor_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_1/test_embedding_vectors_on_epoch_500.npy')
    y_test = np.load('./results/xor_activations_final_rerun/test_embedding_acts/y_test.npy')
    # c_test = np.load('./results/xor_activations_final_rerun/test_embedding_acts/c_test.npy')
    # y_test = np.load('./results/xor_activations_final_rerun/test_embedding_acts/y_test.npy')

    c_test = c_test.reshape((-1, 2, 32))

    fit(c_test, y_test)

    tsne = TSNE(n_components=2, random_state=42)
    x2 = tsne.fit_transform(c_val)

    plt.figure()
    plt.scatter(x2[:, 0], x2[:, 1], marker='o', c='k', alpha=0.1, label='samples')
    plt.scatter(x2[model.centroids[0], 0], x2[model.centroids[0], 1], marker='x', c='r', label='centroids')
    plt.scatter(x2[model.centroids[1], 0], x2[model.centroids[1], 1], marker='x', c='r', label=None)
    plt.scatter(x2[model.centroids[2], 0], x2[model.centroids[2], 1], marker='x', c='r', label=None)
    plt.legend()
    plt.savefig('clustering.png')
    plt.show()

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
