import unittest

import torch
import numpy as np
import unittest
from torch.nn import ModuleList

from torch_explain.logic.commons import Rule, Domain
from torch_explain.logic.grounding import DomainGrounder
from torch_explain.logic.indexing import Indexer
from torch_explain.nn.concepts import ConceptReasoningLayer


class TestTemplateObject(unittest.TestCase):

    def test_R2N_and_DCR(self):
        # data
        emb_size = 4
        X = torch.randn(size=[3, emb_size])
        q_names = ['q(0)', 'q(1)', 'q(2)', 'r(0,1)', 'r(1,2)', 'r(2,0)']
        q_labels = torch.randint(0, 2, size=[6, 1])

        # logic
        points = Domain("points", [f'{i}' for i in np.arange(len(X)).tolist()])
        rule = Rule("phi", body=["r(X,Y)", "q(X)"], head=["q(Y)"], var2domain={"X": "points", "Y": "points"})
        grounder = DomainGrounder({"points": points.constants}, [rule])
        groundings = grounder.ground()
        indexer = Indexer(groundings, q_names)
        indexer.index_all()

        # models
        relation_classifiers = ModuleList([
            torch.nn.Sequential(torch.nn.Linear(emb_size, 1)),  # q(X) classifier
            torch.nn.Sequential(torch.nn.Linear(emb_size*2, 1)),  # r(X,Y) classifier
        ])
        relation_embedders = ModuleList([
            torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size)),  # q(X) classifier
            torch.nn.Sequential(torch.nn.Linear(emb_size*2, emb_size)),  # r(X,Y) classifier
        ])
        task_predictor = ConceptReasoningLayer(emb_size=emb_size*2, n_concepts=2, n_classes=2)

        # relation/concept predictions
        preds_rel, embs_rel = [], []
        for rel_id, (relation_classifier, relation_embedder) in enumerate(zip(relation_classifiers, relation_embedders)):
            embed_tuple, index_tuple, atom_ids = indexer.apply_index(X, 'atoms', rel_id)
            preds_rel.append(relation_classifier(embed_tuple))
            embs_rel.append(relation_embedder(embed_tuple))
        preds_rel = torch.cat(preds_rel, dim=0)
        embs_rel = torch.cat(embs_rel, dim=0)

        # task predictions
        preds_xformula, index_xformula, formula_ids = indexer.apply_index(preds_rel, 'formulas', 0)
        embed_xformula, index_xformula, formula_ids = indexer.apply_index(embs_rel, 'formulas', 0)
        y_preds = task_predictor(embed_xformula, preds_xformula)


if __name__ == '__main__':
    unittest.main()
