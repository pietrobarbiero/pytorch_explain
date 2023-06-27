import torch
import matplotlib.pyplot as plt

from datasets.toy_manifold import manifold_toy_dataset
from model import ManifoldRelationalDCR
import numpy as np

import unittest


def group_by(t, dim):
    ids = t[:, dim].unique()
    mask = t[:, None, dim] == ids
    l = []
    for i, id_rel in enumerate(ids):
        temp = t[torch.argwhere(mask[:, i])]
        temp = temp.squeeze(1)
        l.append(temp)
    return l

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



    def test_tuple_creator_boolean_mask(self):

        X = np.random.randint(size = [ 10, 4], low = 0, high = 1)

        # id_atom, id_rel, id_const, id_pos
        indices = [[0, 42, 0, 0],       #r(a,b) r a 0
                   [0, 42, 1, 1],       #r(a,b) r b 1
                   [1, 42, 0, 0],       #r(a,c) r a 0
                   [1, 42, 2, 1],       #r(a,c) r c 1
                   [2, 43, 0, 0],       #q(a) q a 0
                   [3, 43, 2, 0]]     #q(c) q c 0

        X = torch.tensor(X)
        indices = torch.tensor(indices)



        split_per_rels = group_by(indices, dim=1)
        for sp in split_per_rels:
            print(sp)
            print(torch.stack(group_by(sp, 0),dim=0))

