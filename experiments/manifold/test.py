import torch
import matplotlib.pyplot as plt

from datasets.toy_manifold import manifold_toy_dataset
from model_sudoku import SudokuRelationalDCR
from model import ManifoldRelationalDCR
import numpy as np

import unittest

class ManifoldTest(unittest.TestCase):

    def test_manifold_with_given_relation(self):

        # Only groundings where the relation is true are given
        X, y, body_index, head_index, relation_labels, task_labels = manifold_toy_dataset("moon", only_on_manifold=True)
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
        m = ManifoldRelationalDCR(input_features=2, emb_size=3, manifold_arity=2, num_classes=2, predict_relation=False)

        X = torch.tensor(X, dtype=torch.float)
        y, body_index, head_index, relation_labels, task_labels = (torch.tensor(i) for i in (y, body_index, head_index, relation_labels, task_labels))
        c, t, e = m(X, body_index, head_index)
        # Losses should match the following vectors (now dummy comparison, just for shape)
        return c.shape == torch.eye(2)[y].shape and t.shape == torch.eye(2)[task_labels].shape



    def test_manifold_with_learnable_relation(self):
        # All the groundings are given and we predict the relation as well (and we use it in reasoning)
        X, y, body_index, head_index, relation_labels, task_labels = manifold_toy_dataset("moon", only_on_manifold=False)
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
        m = ManifoldRelationalDCR(input_features=2, emb_size=3, manifold_arity=2, num_classes=2, predict_relation=True)

        X = torch.tensor(X, dtype=torch.float)
        y, body_index, head_index, relation_labels, task_labels = (torch.tensor(i) for i in (y, body_index, head_index, relation_labels, task_labels))
        c, r, t, e = m(X,body_index, head_index)
        # Losses should match the following vectors (now dummy comparison, just for shape)
        return c.shape == torch.eye(2)[y].shape and t.shape == torch.eye(2)[task_labels].shape and r == np.reshape(relation_labels, [-1,1])

    def test_sudoku(self):
        n = 4
        X, y, manifolds, task_labels = sudoku(4)
        m = SudokuRelationalDCR(input_features=28*28, emb_size=10, manifold_arity=n, num_classes=10)

        X = torch.stack(X, dim=0)
        y, manifolds, task_labels = (torch.tensor(i) for i in (y, manifolds, task_labels))
        c, t, e = m(X,manifolds)

        return c.shape == torch.eye(2)[y].shape and t.shape == torch.eye(2)[task_labels].shape
