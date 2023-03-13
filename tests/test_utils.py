import unittest

import torch
# from sklearn.datasets import make_classification
# from sklearn.model_selection import StratifiedShuffleSplit
# from torch.nn.functional import one_hot
# from torch import nn
from torch.nn.functional import one_hot, leaky_relu
# from torch_geometric.nn import Sequential, GCNConv
# from torch_geometric.utils import from_networkx
# import networkx as nx
# import numpy as np

import torch_explain as te
from torch_explain.logic.metrics import test_explanation, complexity, concept_consistency, formula_consistency
from torch_explain.logic.nn import entropy, psi
from torch_explain.nn.functional import prune_equal_fanin


class TestTemplateObject(unittest.TestCase):
    def test_get_predictions(self):
        x, c, y = te.datasets.xor(500)
        formula = '(x1 & x2)'
        te.logic.utils.get_predictions(formula, c, 0.5)
        formula = ''
        te.logic.utils.get_predictions(formula, c, 0.5)


if __name__ == '__main__':
    unittest.main()
