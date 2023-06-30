import torch
import matplotlib.pyplot as plt
import numpy as np
import unittest
from torch.nn import ModuleList

from datasets.toy_manifold import manifold_toy_dataset
from model import ManifoldRelationalDCR

from torch_explain.logic.commons import Rule, Domain
from torch_explain.logic.grounding import DomainGrounder
from torch_explain.logic.indexing import Indexer, group_by_no_for
from torch_explain.nn.concepts import ConceptReasoningLayer


class ManifoldTest(unittest.TestCase):

    def test_R2N_and_DCR(self):
        # data
        emb_size = 4
        X = torch.randn(size=[3, emb_size])
        q_names = ['r(1,2)', 'r(2,0)', 'q(2)', 'q(0)', 'q(1)']
        q_labels = torch.randint(0, 2, size=[5, 1])

        # logic
        points = Domain("points", [f'{i}' for i in np.arange(len(X)).tolist()])
        rule = Rule("phi", body=["r(X,Y)", "q(X)"], head=["q(Y)"], var2domain={"X": "points", "Y": "points"})
        grounder = DomainGrounder({"points": points.constants}, [rule])
        groundings = grounder.ground()
        indexer = Indexer(groundings, q_names)
        indexer.index_all()

        # models
        relation_classifiers = ModuleList([
            torch.nn.Sequential(torch.nn.Linear(emb_size, 1), torch.nn.Sigmoid()),  # q(X) classifier
            torch.nn.Sequential(torch.nn.Linear(emb_size*2, 1), torch.nn.Sigmoid()),  # r(X,Y) classifier
        ])
        relation_embedders = ModuleList([
            torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size)),  # q(X) classifier
            torch.nn.Sequential(torch.nn.Linear(emb_size*2, emb_size)),  # r(X,Y) classifier
        ])
        task_predictor = ConceptReasoningLayer(emb_size=emb_size*2, n_concepts=2, n_classes=2)

        # relation/concept predictions
        preds_rel, embs_rel = [], []
        queries_ids = []
        for rel_id, (relation_classifier, relation_embedder) in enumerate(zip(relation_classifiers, relation_embedders)):
            embed_tuple, constants_index, query_index = indexer.apply_index(X, 'atoms', rel_id)
            queries_ids.append(query_index)
            preds_rel.append(relation_classifier(embed_tuple))
            embs_rel.append(relation_embedder(embed_tuple))
        queries_ids = torch.cat(queries_ids, dim=0)
        preds_rel = torch.cat(preds_rel, dim=0)
        embs_rel = torch.cat(embs_rel, dim=0)

        # task predictions
        preds_xformula, index_xformula, formula_ids = indexer.apply_index(preds_rel, 'formulas', 0)
        embed_xformula, index_xformula, formula_ids = indexer.apply_index(embs_rel, 'formulas', 0)
        y_preds = task_predictor(embed_xformula, preds_xformula)

        # aggregate task predictions (next: do it with OR)
        y_preds_group = group_by_no_for(groupby_values=formula_ids, tensor_to_group=y_preds)
        y_preds_group = torch.stack(y_preds_group, dim=0)
        y_preds_mean = y_preds_group.mean(dim=1)

        # get supervised slice
        sup_preds_rel = indexer.get_supervised_slice(preds_rel, queries_ids)
        sup_y_rel = indexer.get_supervised_slice(q_labels, torch.unique(formula_ids))

        # check predictions
        correct_predictions_rel = (q_labels == (sup_preds_rel>0.5))
        correct_predictions_y = (sup_y_rel.squeeze() == (y_preds_mean[:, 1]>0.5))

        print(correct_predictions_rel)
        print(correct_predictions_y)


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
