import torch
import matplotlib.pyplot as plt

from datasets.toy_manifold import manifold_toy_dataset
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



    def test_tuple_creator_boolean_mask(self):

        def group_by_no_for(t, dim):
            _, indices = torch.sort(t[:, dim])
            t = t[indices]
            ids = t[:, dim].unique()
            mask = t[:, None, dim] == ids
            splits = torch.argmax(mask.float(), dim=0)
            r = torch.tensor_split(t, splits[1:])
            return r

        X = np.random.random(size=[10, 4])

        # id_atom, id_rel, id_const, id_pos
        indices = [[0, 42, 0, 0],  # r(a,b) r a 0
                   [0, 42, 1, 1],  # r(a,b) r b 1
                   [1, 42, 2, 1],  # r(a,c) r c 1
                   [2, 43, 0, 0],  # q(a) q a 0
                   [1, 42, 0, 0],  # r(a,c) r a 0
                   [3, 43, 2, 0]]  # q(c) q c 0

        X = torch.tensor(X)
        indices = torch.tensor(indices)

        # We need them to be sorted first, by relation (column=0) and, then, by position (columsn=3)
        indices = indices[torch.argsort(indices[:, 3])]
        indices = indices[torch.argsort(indices[:, 1], stable=True)]

        split_per_rels = group_by_no_for(indices, dim=1)
        for rel in split_per_rels:
            rel_id = rel[0, 1]
            tuples = group_by_no_for(rel, dim=0)
            tuples = torch.stack(tuples, dim=0)
            atom_ids = tuples[:, 0, 0]
            tuples = tuples[:, :, 2]
            embed_tuple = X[tuples].view(tuples.shape[0], -1)

            print()
            print("Id relation", rel_id)
            print("Id atoms corresponding to tuples", atom_ids)
            print("Id constant in tuples", tuples)
            print("Embedding tuple", embed_tuple)

